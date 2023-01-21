---
title: 'Tiling'
linkTitle: 'tiling'
description: ''
---

Description
-----------

This API allows you to transform a dataset into a tiled one. This transform is known to be useful for detecting small objects in high-resolution input images <a href="#ref1">[1]</a>. In general, we resize the input image to be small enough for the network to afford in terms of memory and computational cost. This practice makes small objects much smaller that the neural network cannot distinguish them. The tiling transform increases the size of a small object's area relative to the network's receptive area to make it distinguishable. Therefore, this transform allows you to easily construct a dataset to train a model to detect small objects in high-resolution images.

References
----------
<ol>
    <li id="ref1">
        F., Ozge Unel, Burak O. Ozkalayci, and Cevahir Cigla. "The power of tiling for small object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2019.
    </li>
</ol>

Jupyter Notebook Example
------------------------
{{< blocks/notebook 06_tiling >}}
