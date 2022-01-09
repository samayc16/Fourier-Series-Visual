# Fourier-Series-Visual
Reconstructs the contours of an input image or string by estimation via a fourier series visually

## What?
A Fourier series is an infinite series of sinusoids used to approximate a function.

## Why?
I wanted to create a project in which I could do something cool with the concepts I learned in my introductory signal processing class. After seeing a 3Blue1Brown video on drawing a single-line portrait via a Fourier series, I had an idea to process any input image to get its contour ordered in a single line, and redrew it in 3 dimensions with a Fourier series.

## How?

In order to approximate f(t) with a single Fourier series, I mapped all of the y-coordinates to the complex number line, thus mapping f(t) to the complex plane rather than the x-y plane. With this, instead of f(t) outputting two values for the position of x and y, it now outputs a single complex quantity containing the information of both x and y.  

Now with f(t) defined, I moved forward with the Fourier coefficients.

A Fourier series is described as ![equation](https://latex.codecogs.com/svg.image?\bg_white&space;\inline&space;f(t)&space;=&space;\lim_{N\to\infty}\sum_{k=&space;-N}^{N}c_ke^{-j\frac{{2\pi}kt}{T}})
