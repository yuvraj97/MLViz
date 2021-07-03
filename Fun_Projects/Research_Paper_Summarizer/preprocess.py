from numpy import ndarray
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from collections import defaultdict
from typing import BinaryIO, Tuple, Dict, Union, List
from math import *
from collections import Counter
import pdfminer
import numpy as np
import nltk

def createPDFDoc(fp: BinaryIO) -> PDFDocument:
    parser = PDFParser(fp)
    document = PDFDocument(parser, password='')
    if not document.is_extractable:
        raise ValueError("Not extractable")
    else:
        return document

def createDeviceInterpreter() -> Tuple[PDFPageAggregator, PDFPageInterpreter]:
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    return device, interpreter

def parse_obj(objs, hlen) -> Dict[int, Dict[int, Dict[str, Union[float, int, str]]]]:
    js: Dict[int, Dict[int, Dict[str, Union[float, int, str]]]]
    js = defaultdict(lambda: defaultdict(lambda: {}))
    for obj in objs:
        # if it's a container, recurse
        if isinstance(obj, pdfminer.layout.LTFigure):
            parse_obj(obj._objs, hlen)
        elif isinstance(obj, pdfminer.layout.LTTextBox):
            for o in obj._objs:
                if isinstance(o, pdfminer.layout.LTTextLine):
                    x0, y0 = int(o.x0), int(o.y0)
                    js[y0 + hlen][x0] = {
                        "bbox": o.bbox,
                        "height": floor(o.height),
                        "width": o.width,
                        "text": o.get_text()
                    }

    return js

def get_side_by_side_text(
        js: Dict[int, Dict[int, Dict[str, Union[float, int, str]]]],
        hgap: int,
        sentence_width: Union[float, ndarray]) -> List[Tuple[str, Tuple[int, int]]]:

    l_sections: List[Tuple[str, Tuple[int, int]]] = [("", (0, np.inf))]
    r_sections: List[Tuple[str, Tuple[int, int]]] = [("", (0, np.inf))]
    for y0 in reversed(sorted(js)):
        for x0 in sorted(js[y0]):
            text = js[y0][x0]["text"].strip()
            if len(text.strip()) <= 1: continue
            if x0 < sentence_width:
                if l_sections[-1][1][1] - y0 > hgap: text = '\n\n' + text.strip()
                l_sections.append((text, (x0, y0)))
            else:
                if r_sections[-1][1][1] - y0 > hgap: text = '\n\n' + text.strip()
                r_sections.append((text, (x0, y0)))
    return l_sections + r_sections

def get_straight_text(
        js: Dict[int, Dict[int, Dict[str, Union[float, int, str]]]],
        hgap: int) -> List[Tuple[str, Tuple[int, int]]]:

    sections: List[Tuple[str, Tuple[int, int]]] = [("", (0, np.inf))]
    for y0 in reversed(sorted(js)):
        for x0 in sorted(js[y0]):
            text = js[y0][x0]["text"].strip()
            if len(text.strip()) <= 1: continue
            if sections[-1][1][1] - y0 > hgap: text = '\n\n' + text.strip()
            sections.append((text, (x0, y0)))
    return sections

def clean_sections(s: str) -> str:
    L = []
    for part in s.split(" "):
        if True in [token in part for token in ["http://", "https://", ".com"]]:
            continue
        L.append(part)
    return " ".join(L)

def processPDF(fp: BinaryIO) -> Dict[str, List[Dict[str, str]]]:
    """

    @param fp: binary file pointer
    @return:
    """

    document = createPDFDoc(fp)  # It will close the file, so no need of fp.close()
    device, interpreter = createDeviceInterpreter()
    pages = PDFPage.create_pages(document)

    js: Dict[int, Dict[int, Dict[str, Union[float, int, str]]]] = {}
    hlen: int = 100000
    for page_no, page in enumerate(pages):
        interpreter.process_page(page)
        layout = device.get_result()
        _js = parse_obj(layout._objs, hlen * layout.height)
        if True in [True for y0 in _js for x0 in _js[y0] if "References" in _js[y0][x0]["text"]]:
            break
        js = {**js, **_js}
        hlen -= 1

    ys: List[int] = sorted(js.keys())
    sentence_hgap: int = Counter([ys[i + 1] - ys[i] for i in range(len(ys) - 1)]).most_common(1)[0][0]
    sentence_font_size: int = Counter([js[y0][x0]["height"] for y0 in js for x0 in js[y0]]).most_common(1)[0][0]
    sentence_width = np.mean([js[y0][x0]["width"] for y0 in js for x0 in js[y0]
                              if js[y0][x0]["height"] == sentence_font_size])
    side_by_side = True if sentence_width < layout.width * 0.65 else False
    if side_by_side: sections = get_side_by_side_text(js, sentence_hgap, sentence_width)
    else: sections = get_straight_text(js, sentence_hgap)
    sentences_len: List[int] = [len(sentence) for text, _ in sections for sentence in nltk.sent_tokenize(text)]
    min_sentence_length = np.mean(sentences_len)
    sectionsL: List[str] = " ".join([text for text, _ in sections]).split("\n\n")
    sectionsL = [" ".join([clean_sections(sentence) for sentence in nltk.sent_tokenize(section)
                           if len(sentence) > min_sentence_length])
                 for section in sectionsL]
    sectionsL = [section for section in sectionsL
                 if section != "" and len(section) > min_sentence_length and len(nltk.sent_tokenize(section)) > 1]

    return {"sections": [{"text": section} for section in sectionsL]}
