//  Copyright (C) 2023 Intel Corporation
//
//  SPDX-License-Identifier: MIT

mod coco_page_mapper;
mod datum_page_mapper;
mod page_mapper;
mod page_maps;
mod test_helpers;
mod utils;
use pyo3::prelude::*;

use crate::coco_page_mapper::CocoPageMapper;
use crate::datum_page_mapper::DatumPageMapper;

/// Datumaro Rust API
#[pymodule]
#[pyo3(name = "rust_api")]
fn rust_api(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CocoPageMapper>()?;
    m.add_class::<DatumPageMapper>()?;

    Ok(())
}
