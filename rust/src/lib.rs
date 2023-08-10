//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

mod coco_page_mapper;
mod page_maps;
mod utils;

use std::{fs::File, io::BufReader, path::Path};

use crate::coco_page_mapper::CocoPageMapper as CocoPageMapperImpl;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyList, PyUnicode},
};
use serde_json;

#[pyclass]
struct CocoPageMapper {
    reader: BufReader<File>,
    mapper: CocoPageMapperImpl,
}

fn convert_to_py_object(value: &serde_json::Value, py: Python<'_>) -> PyResult<PyObject> {
    if value.is_array() {
        let list = PyList::empty(py);

        for child in value.as_array().unwrap() {
            list.append(convert_to_py_object(child, py)?)?;
        }

        return Ok(list.into());
    } else if value.is_object() {
        let dict = PyDict::new(py);

        for (key, child) in value.as_object().unwrap().iter() {
            let child = convert_to_py_object(child, py)?;
            dict.set_item(key, child)?;
        }

        return Ok(dict.into());
    } else if value.is_boolean() {
        return Ok(PyBool::new(py, value.as_bool().unwrap()).into());
    } else if value.is_f64() {
        return Ok(PyFloat::new(py, value.as_f64().unwrap()).into());
    } else if value.is_i64() {
        return Ok(value.as_i64().unwrap().to_object(py));
    } else if value.is_u64() {
        return Ok(value.as_u64().unwrap().to_object(py));
    } else if value.is_string() {
        return Ok(PyUnicode::new(py, value.as_str().unwrap()).into());
    } else if value.is_null() {
        return Ok(PyUnicode::new(py, "null").into());
    } else {
        return Err(PyValueError::new_err("Unknown value type"));
    }
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
        let item_dict = self.mapper.get_item_dict(img_id, &mut self.reader)?;
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

/// Datumaro Rust API
#[pymodule]
#[pyo3(name = "rust_api")]
fn rust_api(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CocoPageMapper>()?;

    Ok(())
}
