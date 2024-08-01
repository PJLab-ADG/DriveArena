
<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<div align="center">
  <!-- <a href=""> -->
  <img src="docs/github-logo.png" alt="Logo" width="550px">
  <br>
  <a href="https://pjlab-adg.github.io/DriveArena/">
    <img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page" style="height:20px;">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv Paper" style="height:20px;">
  </a>
  <hr>
  <img src="docs/boston_thomas_park.gif" style="width: 800px; display: block; margin: 0;" />
  <img src="docs/singapore.gif" style="width: 800px; display: block; margin: 0;" />
  <img src="docs/boston.gif" style="width: 800px; display: block; margin: 0;" />
  <br>

  <p align="left">
    This is the official project repository of the paper <b> DriveArena: A Controllable Generative Simulation Platform for Autonomous Driving </b> and is mainly used for releasing schedules, updating instructions, sharing model weights, and handling issues. 
  </p>
</div>

<!--
> Xuemeng Yang<sup>1\*</sup>, Licheng Wen<sup>1\*</sup>, Yukai Ma<sup>2,1,\*</sup>, Jianbiao Mei<sup>2,1,\*</sup>, Xin Li<sup>3,5,\*</sup>, Tiantian Wei<sup>1,4,\*</sup>, Wenjie Lei<sup>2</sup>, Daocheng Fu<sup>1</sup>, Pinlong Cai<sup>1</sup>, Min Dou<sup>1</sup>, Botian Shi<sup>1,†</sup>, Liang He<sup>5</sup>, Yong Liu<sup>2,†</sup>, Yu Qiao<sup>1</sup> <br>
> <sup>1</sup> Shanghai Artificial Intelligence Laboratory <sup>2</sup> Zhejiang University <sup>3</sup> Shanghai Jiao Tong University <sup>4</sup> Technical University of Munich <sup>5</sup> East China Normal University <br>
> <sup>\*</sup> Equal Contribution <sup>†</sup> Corresponding Authors
-->

------
### 💡 Notice
DriveArena is currently under active development and will be open-sourced soon. 

**If you want to get informed once the code is released, please fill out this <a href="https://forms.gle/AYtQdiZEvCTr2T56A">Google form</a>.**

### :new: Updates
`[2024-07-30]:` We've released the [project page](https://pjlab-adg.github.io/DriveArena/) of DriveArena!  

------

<!-- ABOUT THE PROJECT -->
## :fire: Highlights

<b> DriveArena </b> is a simulation platform that can

* Provide closed-loop high-fidelity testing environments for vision-based driving agents.
* Dynamically control the movement of all vehicles in the scenarios.
* Generate realistic simulations with road networks from any city worldwide.
* Follow a modular architecture, allowing the easy replacement of each module.

<div align="center">
  <img width=600px src="docs/pipeline_2.png">
</div>

The <b>DriveArena</b> is pretrained on nuScenes dataset. All kinds of vision-based driving agents, such as UniAD and VAD, can be combined with <b>DriveArena</b> to evaluate their actual driving performance in closed-loop realistic simulation environments.

<!-- ROADMAP -->
## 📌 Roadmap

- [x]  Demo Website Release
- [ ]  V1.0 Release
    - [ ]  Traffic Manager Code
    - [ ]  World Dreamer
        - [ ]  Inference Code
        - [ ]  Training Code
        - [ ]  Pretrained Weights
    - [ ]  Evaluation Code
- [ ]  Driving Agent Support
    - [ ]  UniAD
    - [ ]  VAD
    - [ ]  LeapAD
- [ ]  Video AutoRegression Dreamer

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We utilized the following repos during development:

* [LimSim++](https://github.com/PJLab-ADG/LimSim/tree/LimSim_plus)
* [MagicDrive](https://github.com/cure-lab/MagicDrive)
* [UniAD](https://github.com/OpenDriveLab/UniAD)

Thanks for their Awesome open-sourced work!

<!-- LICENSE -->
## License

Distributed under the [Apache 2.0 license](./LICENSE).

<!-- CONTACT -->
## Citation

If you find our paper and codes useful, please kindly cite us via:

```bibtex

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
