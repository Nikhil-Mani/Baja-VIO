
#import "@preview/bananote:0.1.2": *
#import "@preview/pergamon:0.7.1": *


#show: note.with(
  title: [Kalman Filter Implementation Notes],
  authors: (([Nikhil Mani], [Johns Hopkins University]),)
)


#outline()

= Introduction

This project is going to (attempt) to apply a Kalman Filter based approach to perform state estimation on the Blue Jay Racing 23xt. The sensors we will be working with are hall effects for wheel velocity, an IMU mounted in the electronics box, and a GoPro with an included IMU.


= State Vector
Our state vector must contain 4 main components:

+ *Position*: Location in 3D Space
  $ bold(x) = vec(x, y, z) $

+ *Velocity*:
  $ bold(dot(x)) = vec(dot(x), dot(y), dot(z)) $
+ *Acceleration*:
  $ bold(dot.double(x)) = vec(dot.double(x), dot.double(y), dot.double(z)) $
+ *Orientation*: roll, pitch, yaw
  $ bold(hat(theta)) = vec(phi, theta, psi) $
+ *Biases*: we must predict biases for both the accelerometer and the gyroscope:
$ bold(b_{"accel"} = vec(b_{"ax"}, b_{"ay"}, b_{az}))
#pagebreak()
Putting this all together, we end up with a $12 x 1$ vector (that I don't want to type out)
$ vec(bold(x), bold(dot(x)), bold(dot.double(x)), bold(hat(theta))) $





// Uncomment this to typeset the bibliography:
// #add-bib-resource(read("bibliography.bib"))
// #print-bananote-bibliography()
