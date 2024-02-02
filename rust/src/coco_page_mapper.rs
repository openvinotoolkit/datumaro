//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use std::{
    io::{self, Read, Seek},
    str::FromStr,
};
use strum::EnumString;

use crate::{
    page_mapper::{JsonPageMapper, ParsedJsonSection},
    page_maps::{AnnPageMap, ImgPageMap, JsonDict},
    utils::{convert_to_py_object, invalid_data, parse_serde_json_value, read_skipping_ws},
};
use pyo3::{prelude::*, types::PyList};
use std::{fs::File, io::BufReader, path::Path};

#[derive(EnumString, Debug)]
enum CocoJsonSection {
    #[strum(ascii_case_insensitive)]
    LICENSES(JsonDict),
    #[strum(ascii_case_insensitive)]
    INFO(JsonDict),
    #[strum(ascii_case_insensitive)]
    CATEGORIES(JsonDict),
    #[strum(ascii_case_insensitive)]
    IMAGES(ImgPageMap<i64>),
    #[strum(ascii_case_insensitive)]
    ANNOTATIONS(AnnPageMap),
}

impl ParsedJsonSection for CocoJsonSection {
    fn parse(
        buf_key: String,
        mut reader: impl Read + Seek,
    ) -> Result<Box<CocoJsonSection>, io::Error> {
        match CocoJsonSection::from_str(buf_key.as_str()) {
            Ok(curr_key) => {
                while let Ok(c) = read_skipping_ws(&mut reader) {
                    if c == b':' {
                        break;
                    }
                }
                match curr_key {
                    CocoJsonSection::LICENSES(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(CocoJsonSection::LICENSES(v)))
                    }
                    CocoJsonSection::INFO(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(CocoJsonSection::INFO(v)))
                    }
                    CocoJsonSection::CATEGORIES(_) => {
                        let v = parse_serde_json_value(reader)?;
                        Ok(Box::new(CocoJsonSection::CATEGORIES(v)))
                    }
                    CocoJsonSection::IMAGES(_) => {
                        let v = ImgPageMap::from_reader(reader)?;
                        Ok(Box::new(CocoJsonSection::IMAGES(v)))
                    }
                    CocoJsonSection::ANNOTATIONS(_) => {
                        let v = AnnPageMap::from_reader(reader)?;
                        Ok(Box::new(CocoJsonSection::ANNOTATIONS(v)))
                    }
                }
            }
            Err(e) => {
                let cur_pos = reader.stream_position()?;
                let msg = format!("Unknown key: {} at pos: {}", e, cur_pos);
                Err(invalid_data(msg.as_str()))
            }
        }
    }
}

#[derive(Debug)]
pub struct CocoPageMapperImpl {
    licenses: JsonDict,
    info: JsonDict,
    categories: JsonDict,
    images: ImgPageMap<i64>,
    annotations: AnnPageMap,
}

impl JsonPageMapper<CocoJsonSection> for CocoPageMapperImpl {}

impl CocoPageMapperImpl {
    pub fn licenses(&self) -> &JsonDict {
        return &self.licenses;
    }
    pub fn info(&self) -> &JsonDict {
        return &self.info;
    }
    pub fn categories(&self) -> &JsonDict {
        return &self.categories;
    }
    pub fn get_img_ids(&self) -> &Vec<i64> {
        self.images.ids()
    }
    pub fn get_item_dict(
        &self,
        img_id: &i64,
        mut reader: impl Read + Seek,
    ) -> Result<JsonDict, io::Error> {
        self.images.get_dict(&mut reader, img_id)
    }
    pub fn get_anns_dict(
        &self,
        img_id: i64,
        mut reader: impl Read + Seek,
    ) -> Result<Vec<JsonDict>, io::Error> {
        self.annotations.get_anns(&mut reader, img_id)
    }
    pub fn new(mut reader: impl Read + Seek) -> Result<Self, io::Error> {
        let sections = Self::parse_json(&mut reader)?;

        let mut licenses = None;
        let mut info = None;
        let mut categories = None;
        let mut images = None;
        let mut annotations = None;

        for section in sections {
            match *section {
                CocoJsonSection::LICENSES(v) => {
                    licenses = Some(v);
                }
                CocoJsonSection::INFO(v) => {
                    info = Some(v);
                }
                CocoJsonSection::CATEGORIES(v) => {
                    categories = Some(v);
                }
                CocoJsonSection::IMAGES(v) => {
                    images = Some(v);
                }
                CocoJsonSection::ANNOTATIONS(v) => {
                    annotations = Some(v);
                }
            }
        }

        let licenses = licenses.ok_or(invalid_data("Cannot find the licenses section."))?;
        let info = info.ok_or(invalid_data("Cannot find the info section."))?;
        let categories = categories.ok_or(invalid_data("Cannot find the categories section."))?;
        let images = images.ok_or(invalid_data("Cannot find the images section."))?;
        let annotations =
            annotations.ok_or(invalid_data("Cannot find the annotations section."))?;

        Ok(CocoPageMapperImpl {
            licenses,
            info,
            categories,
            images,
            annotations,
        })
    }
}

#[pyclass]
pub struct CocoPageMapper {
    reader: BufReader<File>,
    mapper: CocoPageMapperImpl,
}

#[pymethods]
impl CocoPageMapper {
    #[new]
    fn py_new(path: String) -> PyResult<Self> {
        let file = File::open(Path::new(&path))?;
        let mut reader = BufReader::new(file);
        let mapper = CocoPageMapperImpl::new(&mut reader)?;

        Ok(CocoPageMapper { reader, mapper })
    }

    fn licenses(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.licenses(), self_.py())
    }

    fn info(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.info(), self_.py())
    }

    fn categories(self_: PyRef<Self>) -> PyResult<PyObject> {
        convert_to_py_object(self_.mapper.categories(), self_.py())
    }

    fn get_item_dict(&mut self, py: Python<'_>, img_id: i64) -> PyResult<PyObject> {
        let item_dict = self.mapper.get_item_dict(&img_id, &mut self.reader)?;
        Ok(convert_to_py_object(&item_dict, py)?)
    }

    fn get_anns_dict(&mut self, py: Python<'_>, img_id: i64) -> PyResult<PyObject> {
        let anns_list = PyList::new(
            py,
            self.mapper
                .get_anns_dict(img_id, &mut self.reader)?
                .iter()
                .map(|child| convert_to_py_object(child, py).unwrap()),
        );
        Ok(anns_list.into())
    }

    fn get_img_ids(&self) -> Vec<i64> {
        self.mapper.get_img_ids().to_owned()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.mapper.get_img_ids().len())
    }
}
