# Fourier-Series-Visual
Reconstructs the contours of an input image or string by estimation via a fourier series visually

## What?
A Fourier series is an infinite series of sinusoids used to approximate a function. I created a program in which the user inputs an image, and the outline is redreawn visually via a Fourier Series.

## Why?
I wanted to create a project in which I could do something cool with the concepts I learned in my introductory signal processing class. After seeing a 3Blue1Brown video on drawing a single-line portrait via a Fourier series, I had an idea to process any input image to get its contour ordered in a single line, and redrew it in 3 dimensions with a Fourier series.

## How?

In order to approximate f(t) with a single Fourier series, I mapped all of the y-coordinates to the complex number line, thus mapping f(t) to the complex plane rather than the x-y plane. With this, instead of f(t) outputting two values for the position of x and y, it now outputs a single complex quantity containing the information of both x and y.  

Now with f(t) defined, I moved forward with the Fourier coefficients.

A Fourier series is described as  ![equation](https://latex2png.com/pngs/5318e50ad64246ca9d882dfd06843284.png) , where *c<sub>k</sub>* denotes the Fourier coefficient for each term. The Fourier coefficient is calculated by  ![equation](https://latex2png.com/pngs/49ed4add9ffbb93e3e5bc03bac0a4ae5.png). The Fourier coefficient calculates the average amplitude and phase of a complex sinusoid with frequency k.

The closer N, the order of the Fourier series approximation, is to infinity, the closer the approximation is to the inputted image. After a user inputs an image, the order they want to the approximation to be, and the fps of their desired video, the program calculates the Fourier Series coefficients and draws the point of the Fourier series over time. For more information, check out the iPython Notebook explaining the code!
