# State Estimation Implementation for Baja Car
This repository will contain my implementation of state estimation through an ESKF (and later factor graphs) for the JHU Baja Team.


### Current Progress
Inertial portion of the ESKF is completed, yielding strong results on testing data although showing some drift in the z-direction due to a lack of measurements.
The next step is to create a VIO system combining the ESKF with the GoPro camera and later a fully integrated factor graph, which may yield better performance.
