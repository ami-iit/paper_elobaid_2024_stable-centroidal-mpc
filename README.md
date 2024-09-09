<h1 align="center">
Online Non-linear Centroidal MPC with Stability Guarantees
for Robust Legged Robots Locomotion</h1>

<div align="center">
<p>Mohamed Elobaid, Giulio Turrisi, Lorezno Rapetti, Giulio Romualdi, Stefano Dafarra, </br>
Tomohiro Kawakami, Tomohiro Chaki, Takahide Yoshiike, Claudio Semini and Daniele Pucci</p>
</div>

<br>


<div align="center">
    📅 Submitted to the IEEE Robotics and Automation Letters (RAL) 🤖
</div>

<div align="center">
    <a href="#Code-organization"><b>🔧 Code 🔧</b></a>
    <a href="#Examples-and-usage"><b>🔧 Examples 🔧</b></a>
</div>


## Paper

The paper is under review. The preprint is available in the following link https://arxiv.org/pdf/2409.01144


https://github.com/user-attachments/assets/1869417f-f94b-4368-861f-c0f3b415bb02



## Code organization

The MPC problem formulation for both the humanoid case and the quadruped case are housed in submodules. The code is organized as follows
-  examples houses example code and scripts allowing to install and run the MPC and produce plots/simulations for a bipedal and a quadruped.
- `humanoid-mpc` points to the implementation of the Centroidal MPC (Casadi + IPOPT) with the possibility of setting a flag enabling or disabling the stability constraints
- `quadruped-mpc` points to the implementation of the Centroidal MPC through ACADOS with the possibility of using different quadruped robots and terrains.

## Examples and usage
To use the examples in the example directory you can simply run the corresponding `run_and_plot.sh` script as in the following 
### humanoid example
The humanoid example simulates using the Centroidal MPC with the flag `enable_stability_cstr` off and then on. It then generates gifs/animation of the data of the footsteps and CoM trajectory in both cases.
The simulation, to test for edge cases, runs the MPC with `dt = 200ms` with a horizon length of `5 steps`. In this way we emphasize the role played by the stability constraints in ensuring success as opposed to the nominal case. To run the example you simply do

```bash
cd examples/humanoid
chmod +x run_and_plot.sh
./run_and_plot.sh
```
The above script builds the MPC suite housed in the submodule `humanoid-mpc` . Make a build directory for the example in `main.cpp` , build and install it, and then run the matlab plotting scripts for the generated data.
If successful, you should be able to see animations of this kind in your build/bin directory

<table>
    <tr>
        <th colspan="2" align="center">Simulations without (left) and With (right) Stability for a biped</th>
    </tr>
    <tr>
        <td align="left"><img src="https://github.com/user-attachments/assets/01f02201-4a7e-4782-9a94-f5670cb887ec" alt="Without Stability" /></td>
        <td align="center"><img src="https://github.com/user-attachments/assets/55e59f28-07d3-4bb8-adfc-9a8bf36ed847" alt="With Stability" /></td>
    </tr>
</table>


<p> ⚠️ The `run_and_plot.sh` script assumes you have the dependencies of the `bipedallocomotionframework` housed in the submodule `humanoid-mpc`. Please check https://github.com/ami-iit/bipedal-locomotion-framework/?tab=readme-ov-file#page_facing_up-mandatory-dependencies for the details </p>


<p> ⚠️ Do not build `bipedallocomotioninterface` from master. The script will handle building and installing the correct version with the relevant commit. Only ensure you have the dependencies installed in your system </p>


### Quadruped example
The quadruped example simulates using the MPC with and without the stability constraints with a quadruped (Aliengo). It produces the simulations using mojoco. To run the example you simply do

```bash
cd examples/humanoid
chmod +x run_and_plot.sh
./run_and_plot.sh
```

You should be able to see something similar to (with obvious difference in performance between nominal and stable centroidal MPC)


https://github.com/user-attachments/assets/bc05ba45-fcff-4a21-9281-8ed78c6ea37a

## Citing this work

Feel free to open issues on this repo for specific questions. If you find this work useful, consider citing it

```
@article{elobaid2024online,
  title={Online Non-linear Centroidal MPC with Stability Guarantees for Robust Locomotion of Legged Robots},
  author={Elobaid, Mohamed and Turrisi, Giulio and Rapetti, Lorenzo and Romualdi, Giulio and Dafarra, Stefano and Kawakami, Tomohiro and Chaki, Tomohiro and Yoshiike, Takahide and Semini, Claudio and Pucci, Daniele},
  journal={arXiv preprint arXiv:2409.01144},
  year={2024}
}

```

## Maintainers

<table align="left">
    <tr>
        <td><a href="https://github.com/mebbaid"><img src="https://github.com/mebbaid.png" width="40"></a></td>
        <td><a href="https://github.com/mebbaid">👨‍💻 @Mohamed Elobaid</a></td>
    </tr>
</table>
