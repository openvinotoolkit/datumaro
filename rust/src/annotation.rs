//  Copyright (C) 2024 Intel Corporation
//
//  SPDX-License-Identifier: MIT

fn get_shoelace_area(points: Vec<(f64, f64)>) -> f64 {
    let n = points.len();
    // Not a polygon
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % n]; // Next vertex, wrapping around using modulo
        area += x1 * y2 - y1 * x2;
    }
    area.abs() / 2.0
}
