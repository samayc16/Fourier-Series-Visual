import matplotlib.pyplot as plt
import matplotlib.animation as animate
import numpy as np
import cv2 as cv
from scipy import integrate
from numpy import pi


## User Inputs ##
T = 15  # period
N = 1000  # order
FPS = 60  # fps of video
Speed = 1  # speed
Bitrate = 10 ** 4  # Bitrate


## Converts Image to grayscale and provides xlist and ylist ##
# Converting into grayscale
im = cv.imread("website.jpg")
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = 0
for i in range(len(contours)):
    if len(contours[i]) > len(contours[cnt]):
        cnt = i

# Gets contour into x and y components
xlist = np.array(contours[cnt][:, :, 0]).flatten()
ylist = -np.array(contours[cnt][:, :, 1]).flatten()

# Centers image at (0, 0)
xlist = xlist - (np.max(xlist) + np.min(xlist)) / 2
ylist = ylist - (np.max(ylist) + np.min(ylist)) / 2

# Creates plot and plots translucent outline of image
fig, ax = plt.subplots()
ax.plot(xlist, ylist, color=(0, 0, 0, 0.10))


## Create f[n] => f(t) ##
# Creating f[n] as a complex exponentional
f_n = xlist + 1j * ylist

# Returns the linear approximation given two points at time
def f(t):
    n = t / T * len(xlist)
    return np.interp(n, range(len(xlist)), f_n, period=len(xlist))


## f(t) => f(ω) ##
# Fourier Coefficients
coefficients = []  # matrix to store coefficients
for n in range(-N, N + 1):  # orders in c_0, c_1, c_-1, ... c_n, c_-n
    integrand = lambda t: f(t) * np.exp(-1j * (2 * pi / T) * t * n)
    coefficients.append(
        (1 / T) * integrate.quad_vec(integrand, 0, T, limit=2000, full_output=True)[0]
    )
    # def integrand_real(t): return np.real(f(t) * np.exp(-1j*(2*pi/T)*t*n))
    # def integrand_imag(t): return np.imag(f(t) * np.exp(-1j*(2*pi/T)*t*n))
    # coefficients.append(
    #     (1/T)*integrate.quad(integrand_real, 0, T, limit=100, full_output=True)[0] +
    #     (1j/T)*integrate.quad(integrand_imag, 0, T, limit=100, full_output=True)[0])
    print(
        str(round(((len(coefficients) - 1) / (2 * N)) * 100, 3))
        + "% Done with coefficients..."
    )

# Creating circles and lines
# f(ω) => f(t)
circles = [
    ax.plot([], [], color=(0, 0, 0, 0.25), linestyle="solid")[0]
    for i in range(2 * N + 1)
]  # stores circles drawn each frame
lines = [
    ax.plot([], [], color=(0, 0, 0, 0.25), linestyle="solid")[0]
    for i in range(2 * N + 1)
]  # stores lines drawn each frame
ftx, fty = [], []  # stores points' components drawn
drawn_points = ax.plot(ftx, fty, color=(1, 1, 1, 1), linestyle="solid", linewidth=0.5)[
    0
]  # saves points drawn by tip
current_point = ax.plot([], [], color=(1, 49 / 255, 49 / 255, 0.83))[0]
theta = np.linspace(0, 2 * pi, 200)

# Create function to animate each frame
def draw_f(t):  # function to draw frames of animation
    current_x, current_y = np.real(coefficients[N]), np.imag(coefficients[N])
    for n in range(1, N + 1):
        current = coefficients[N + n] * np.exp(1j * (2 * pi / T) * t * n)
        r = np.absolute(current)  # radius of circle to be drawn
        circles[2 * n - 1].set_data(
            current_x + r * np.cos(theta), current_y + r * np.sin(theta)
        )
        lines[2 * n - 1].set_data(
            [current_x, current_x + np.real(current)],
            [current_y, current_y + np.imag(current)],
        )
        current_x, current_y = current_x + np.real(current), current_y + np.imag(
            current
        )
        current = coefficients[N - n] * np.exp(-1j * (2 * pi / T) * t * n)
        r = np.absolute(current)  # radius of circle to be drawn
        circles[2 * n].set_data(
            current_x + r * np.cos(theta), current_y + r * np.sin(theta)
        )
        lines[2 * n].set_data(
            [current_x, current_x + np.real(current)],
            [current_y, current_y + np.imag(current)],
        )
        current_x, current_y = current_x + np.real(current), current_y + np.imag(
            current
        )

    ftx.append(current_x)
    fty.append(current_y)
    drawn_points.set_data(ftx, fty)
    current_point.set_data([current_x], [current_y])
    print(str(round(((t + T) / (2 * T)) * 100, 3)) + "% Done with frames...")
    return [ax]


## Plotting and Saving ##
# Tidies up graph
ax.set_xlim(min(xlist) - 200, max(xlist) + 200)
ax.set_ylim(min(ylist) - 400, max(ylist) + 400)
ax.set_aspect(1)
ax.set_axis_off()

# Saves to video
Writer = animate.writers["ffmpeg"]
writer = Writer(fps=FPS, metadata=dict(artist="Samay Chandna"), bitrate=Bitrate)
print("Endcoding started")
time = np.linspace(-T, T, round(2 * T * FPS * (Speed ** -1)))
anim = animate.FuncAnimation(fig, draw_f, frames=time, interval=5)
anim.save(
    "website.mp4",
    writer=writer,
    savefig_kwargs=dict(facecolor=(0, 0, 0, 0), edgecolor="none"),
)
print("-------------------------------\nCompleted: website.mp4")