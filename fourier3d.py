import matplotlib.pyplot as plt
import matplotlib.animation as animate
import numpy as np
import cv2 as cv
from scipy import integrate
from numpy import pi


## User Inputs ##
T = 20  # period
N = 4  # order
FPS = 60  # fps of video


## Converts Image to grayscale and provides xlist and ylist ##
# Converting into grayscale
im = cv.imread("woman.jpg")
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = 0
for i in range(len(contours)):
    if len(contours[i]) > len(contours[cnt]):
        cnt = i
print("Image processed")

# Gets contour into x and y components
xlist = np.array(contours[cnt][:, :, 0]).flatten()
ylist = -np.array(contours[cnt][:, :, 1]).flatten()

# Centers image at (0, 0)
xlist = xlist - (np.max(xlist) + np.min(xlist)) / 2
ylist = ylist - (np.max(ylist) + np.min(ylist)) / 2

## x[n], y[n] => x(t), y(t) ##
# Allows for interpolation of contours for a continours function and aliasing (non-integer index, short period/low FPS)
def x(t):
    n = t / T * len(xlist)
    return np.interp(n, range(len(xlist)), xlist, period=len(xlist))


def y(t):
    n = t / T * len(ylist)
    return np.interp(n, range(len(ylist)), ylist, period=len(xlist))


### Plotting Time ###

## Plotting Real Image ##
# Creates common frames
time = np.linspace(0, T, T * FPS)

# Create figure and first subplot
fig = plt.figure()

# Plots image and time marker
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax1.plot(x(time), y(time), time)
drawn_real_point = ax1.plot(
    [x(0)], [y(0)], [0], color=(1, 49 / 255, 49 / 255, 0.83), marker="."
)[0]

# Shows f(t), the real map of the contour over time
def update_real_point(t):
    drawn_real_point.set_data_3d(x(t), y(t), t)


# Sets real image's view
ax1.view_init(90, -90)
ax1.grid(False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])


## f[n] => f(t) ##
# Creating f[n] as a complex exponentional
f_n = xlist + 1j * ylist

# Allows for interpolation of contours for a continours function and aliasing (non-integer index, short period/low FPS)
def f(t):
    n = t / T * len(xlist)
    return np.interp(n, range(len(xlist)), f_n, period=len(xlist))


# Store fourier coefficients
coefficients = [2 +2j, 3-4j, -2+8j, 9-4j, -3+2j, 4-8j, -8-9j, -9-9j, 4-2j]  # array to store coefficients
# for n in range(-N, N + 1):  # orders in c₀, c₁, c₋₁, ... cₙ, c_₋ₙ
#     integrand = lambda t: f(t) * np.exp(-1j * (2 * pi / T) * t * n)
#     coefficients.append(
#         (1 / T) * integrate.quad_vec(integrand, 0, T, limit=500, full_output=True)[0]
#     )
#     # def integrand_real(t): return np.real(f(t) * np.exp(-1j*(2*pi/T)*t*n))
#     # def integrand_imag(t): return np.imag(f(t) * np.exp(-1j*(2*pi/T)*t*n))
#     # coefficients.append(
#     #     (1/T)*integrate.quad(integrand_real, 0, T, limit=100, full_output=True)[0] +
#     #     (1j/T)*integrate.quad(integrand_imag, 0, T, limit=100, full_output=True)[0])
#     print(
#         str(round(((len(coefficients) - 1) / (2 * N)) * 100, 3))
#         + "% Done with coefficients..."
#     )

## f(t) => Re{f(t)}, Im{f(t)} ##
# Plots for X and Y components of Fourier series are made
ax2 = fig.add_subplot(2, 3, 4)
ax3 = fig.add_subplot(2, 3, 5)

# Creates X components, Y components, and resultant sums
fs_xlist, fs_ylist = [], []
sum_x, sum_y = 0, 0
for n in range(-N, N + 1):
    r = np.absolute(coefficients[n + N])
    phi = np.angle(coefficients[n + N])
    sum_x += r * np.cos((2 * pi * n / T) * time + phi)
    sum_y += r * np.sin((2 * pi * n / T) * time + phi)
    fs_xlist.append(
        ax2.plot(
            time,
            r * np.cos((2 * pi * n / T) * time + phi),
            color=((np.abs(n / N), 0, 1 - np.abs(n / N),.25 + 0.25 * np.abs(n / N))),
        )[0]
    )
    fs_ylist.append(
        ax3.plot(
            time,
            r * np.sin((2 * pi * n / T) * time + phi),
            color=((np.abs(n / N), 0, 1 - np.abs(n / N), .25 + 0.25 * np.abs(n / N))),
        )[0]
    )

# Creates vertical line to track F⁻¹{F{f(t)}} (Approximation of f(t) via our Fourier Series)
time_line_1, time_line_2 = ax2.axvline(x=0), ax3.axvline(x=0)


def update_time_line(t):
    time_line_1.set_xdata([t, t])
    time_line_2.set_xdata([t, t])
    return time_line_1, time_line_2


# Sets components' view
ax2.plot(time, sum_x)
ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])

ax3.plot(time, sum_y)
ax3.grid(False)
ax3.set_xticks([])
ax3.set_yticks([])

## Error Plotting Prep##
ax4 = fig.add_subplot(2, 3, 6)
ax4.set_xlim(-T / 2, T + T / 2)
ax4.set_ylim(-1, 5)
error = []
error_plot = ax4.plot([], [])[0]
zero_line = ax4.plot(
    time, np.zeros(len(time)), color=(0, 0, 0, 0.5), linestyle="dashed"  # 0% error line
)

one_line = ax4.plot(
    time, np.ones(len(time)), color=(1, 0, 0, 0.5), linestyle="dashed"
)  # 100% error line

## F⁻¹{F{f(t)}} (Approximation of f(t) via our Fourier Series) ##
ax5 = fig.add_subplot(2, 3, 2, projection="3d")
spirals = [
    ax5.plot(
        [],
        [],
        [],
        linestyle="solid",
        color=(
            (0.5 * np.abs(i / N), 0, 1 - 0.5 * np.abs(i / N), .25 + 0.25 * np.abs(i / N))
        ),
        linewidth=1 - 0.35 * i / N,
    )[0]
    for i in range(2 * N + 1)
]  # stores spirals drawn each frame
lines = [
    ax5.plot(
        [],
        [],
        [],
        linestyle="solid",
        color=((0.5 * np.abs(i / N), 0, 1 - 0.5 * np.abs(i / N), .25 + 0.25 * np.abs(i / N))),
        linewidth=1 - 0.35 * i / N,
    )[0]
    for i in range(2 * N + 1)
]  # stores lines drawn each frame
drawn_fs_point = ax5.plot([], [], [], color=(0, 1, 0, 0.5), marker=".")[
    0
]  # Draws current point
drawn_x, drawn_y, drawn_z = [], [], []  # Stores points of drawn points over time
drawn_fs = ax5.plot([], [], [], color=(0, 0, 0, 0.5))[
    0
]  # Plot of drawn points over time

ax6 = fig.add_subplot(
    2, 3, 3, projection="3d"
)  # Creates plot for only Fourier Series Approximation
fs_only = ax6.plot([], [], [])[0]  # Stores points of Fourier
fs_only_point = ax6.plot([], [], [], color=(0, 1, 0, 0.5), marker=".")[
    0
]  # Draws only current point

# Draws Fourier Series, Approximation, and Error
def draw_f(t):
    current_x, current_y = np.real(coefficients[N]), np.real(
        coefficients[N]
    )  # Offset of c₀
    # Draws spirals, lines, drawing, current point, and error in each frame
    for n in range(1, N + 1):
        current = coefficients[N + n] * np.exp(
            1j * (2 * pi / T) * t * n
        )  # calculates current term
        r = np.absolute(current)  # amplitude of complex sinusoid (spiral)
        phi = np.angle(current)  # phase shift of complex sinusoid (spiral)
        # plots entire period of complex sinusoid
        spirals[2 * n - 1].set_data_3d(
            current_x + r * np.cos((2 * pi * n / T) * time + phi),
            current_y + r * np.sin((2 * pi * n / T) * time + phi),
            time,
        )
        # plots line only at the current time
        lines[2 * n - 1].set_data_3d(
            [current_x, current_x + np.real(current)],
            [current_y, current_y + np.imag(current)],
            [t, t],
        )
        current_x, current_y = current_x + np.real(current), current_y + np.imag(
            current
        )
        current = coefficients[N - n] * np.exp(-1j * (2 * pi / T) * t * n)
        r = np.absolute(current)
        phi = np.angle(current)
        spirals[2 * n].set_data_3d(
            current_x + r * np.cos((2 * pi * -n / T) * time + phi),
            current_y + r * np.sin((2 * pi * -n / T) * time + phi),
            time,
        )
        lines[2 * n].set_data_3d(
            [current_x, current_x + np.real(current)],
            [current_y, current_y + np.imag(current)],
            [t, t],
        )
        current_x, current_y = current_x + np.real(current), current_y + np.imag(
            current
        )
    # After the final point is calculated (the sum of the Fourier series approximation at time t)
    # the image is traced and the current point is drawn as well as the
    drawn_fs_point.set_data_3d(current_x, current_y, t)
    fs_only_point.set_data_3d(current_x, current_y, t)
    drawn_x.append(current_x)
    drawn_y.append(current_y)
    drawn_z.append(t)
    drawn_fs.set_data_3d(drawn_x, drawn_y, drawn_z)
    fs_only.set_data_3d(drawn_x, drawn_y, drawn_z)
    error.append(np.abs((f(t) - (current_x + 1j * current_y)) / f(t)))
    error_plot.set_data(drawn_z, error)
    return ax5


# Graph views are set for FS App, FS Drawing, and Error

ax5.view_init(90, -90)
ax5.axes.set_xlim3d(left=-max(xlist) - 200, right=max(xlist) + 200)
ax5.axes.set_ylim3d(bottom=-max(ylist) - 200, top=max(ylist) + 200)
ax5.axes.set_zlim3d(bottom=-T / 2, top=T + T / 2)


ax6.view_init(90, -90)
ax6.axes.set_xlim3d(left=np.min(xlist) - 200, right=np.max(xlist) + 200)
ax6.axes.set_ylim3d(bottom=np.min(ylist) - 200, top=np.max(ylist) + 200)
ax6.axes.set_zlim3d(bottom=-T / 2, top=T + T / 2)

ax4.grid(False)
ax4.set_xticks([])
ax4.set_yticks([])

ax5.grid(False)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_zticks([])

ax6.grid(False)
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_zticks([])

## Plot tidying ##
# Shows all plots
fig.set_facecolor((16 / 255, 14 / 255, 255 / 255, 0.45))
ax1.set_facecolor((0, 0, 0, 0))
ax2.set_facecolor((0, 0, 0, 0))
ax3.set_facecolor((0, 0, 0, 0))
ax4.set_facecolor((0, 0, 0, 0))
ax5.set_facecolor((0, 0, 0, 0))
ax6.set_facecolor((0, 0, 0, 0))

ax1.set_title("Contour of Original Image")
ax1.set_xlabel("x position of point")
ax1.set_ylabel("y position of point")
ax1.set_zlabel("Time (sec)")

ax2.set_title("X-components of Fourier Series")
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("Amplitude of X Component")

ax3.set_title("Y-components of Fourier Series")
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Amplitude of Y Component")

ax4.set_title("%" + "Error of Fourier Series vs Original")
ax5.set_xlabel("Time (sec)")
ax5.set_ylabel("%" + "Error (Red = 100%, Grey = 0%)")

ax5.set_title("Fourier Series Approximation of Contours")
ax5.set_xlabel("x axis")
ax5.set_ylabel("y axis")
ax5.set_zlabel("Time (sec)")

ax6.set_title("Output of Fourier Series")
ax6.set_xlabel("x position of point")
ax6.set_ylabel("y position of point")
ax6.set_zlabel("Time (sec)")

## Animation##
# Create animation function ojects
real_animation = animate.FuncAnimation(fig, update_real_point, frames=time)
component_animation = animate.FuncAnimation(fig, update_time_line, frames=time)
fs_animation = animate.FuncAnimation(fig, draw_f, frames=time)

# animate
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
plt.show()
