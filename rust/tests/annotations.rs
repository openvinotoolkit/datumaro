//  Copyright (C) 2024 Intel Corporation
//
//  SPDX-License-Identifier: MIT

use datumaro_rust_api::annotations::get_shoelace_area;

#[test]
fn test_get_shoelace_area() {
    // Define a polygon example
    let points = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];

    // Calculate the shoelace area
    let area = get_shoelace_area(&points);

    // Assert that the calculated area matches the expected area (12.0 in this case)
    assert_eq!(area, 12.0);
}
