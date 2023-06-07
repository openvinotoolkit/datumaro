// Copyright (c) 2022, Intel Corporation
// Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.

#include <cstdint>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef double *BB;
struct RLE
{
    RLE(siz h, siz w, siz m, std::unique_ptr<std::vector<uint>> &_cnts)
        : h(h), w(w), m(m), cnts(std::move(_cnts))
    {
    }
    siz h, w, m;
    std::unique_ptr<std::vector<uint>> cnts;
};

RLE rleEncode(const byte *M, siz h, siz w)
{
    siz j, k, a = w * h;
    uint c;
    byte p;
    auto cnts = std::make_unique<std::vector<uint>>(a + 1);

    const byte *T = M;
    k = 0;
    p = 0;
    c = 0;
    for (j = 0; j < a; j++)
    {
        if (T[j] != p)
        {
            (*cnts)[k++] = c;
            c = 0;
            p = T[j];
        }
        c++;
    }
    (*cnts)[k++] = c;

    return RLE(h, w, k, cnts);
}

py::dict pyRleEncode(py::array_t<std::uint8_t, py::array::f_style | py::array::forcecast> mask)
{
    const auto buf = mask.request();
    if (mask.ndim() != 2)
        throw std::runtime_error("mask should be 2d array.");

    const siz h = buf.shape[0];
    const siz w = buf.shape[1];
    const auto rle = rleEncode(static_cast<const byte *>(buf.ptr), h, w);

    py::array_t<uint> cnts(rle.m);
    auto r = cnts.mutable_unchecked();
    for (siz i = 0; i < rle.m; i++)
        r[i] = (*rle.cnts)[i];

    py::list size(2);
    size[0] = h;
    size[1] = w;

    py::dict dict;
    dict["counts"] = cnts;
    dict["size"] = size;

    return dict;
}

PYBIND11_MODULE(_capi, m)
{
    m.def("encode",
          &pyRleEncode,
          "A function to encode 2D binary mask with uncompressed run-length encoding (RLE).",
          py::arg("mask"));
}
