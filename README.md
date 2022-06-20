# MPC-Inverted-pendulum-in-gazebo


### Run MPC example
```bash
python src/do_mpc_example/src/double_inverted_pendulum.py 
```

### Run Gazebo sim
```bash
roslaunch dual_gazebo dual_drive_gazebo.launch
```

### Run MPC control robot in gazebo 
```bash
python src/dual_gazebo/src/mpc_control.py  
```

<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/174507798-d72e95b2-fcf7-40b3-8943-e8c6f8f3371a.gif"/>
</p>
<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/67572161/174508042-3d83cd4d-5279-4248-966b-5f5bb4009e2d.gif"/>
</p>

