# InvReg
Invariant Feature Regularization for Fair Face Recognition (ICCV'23)

![Image text](https://github.com/PanasonicConnect/InvReg/blob/main/imgs/Picture1.png)

## Requirements
* numpy (1.21.4)
* easydict (1.9)
* onnx (1.10.1)
* scipy (1.6.3)
* tdqm (4.62.3)
* torch (1.11.0)
* mxnet (1.9.1)
* torchvision (0.11.0)
* sklearn (0.24.0)
* python (3.8.12)

## Citation
If you find InvReg helpful in your research, please consider citing: 
```bibtex   
@inproceedings{ma2023invariant,
  title={Invariant Feature Regularization for Fair Face Recognition},
  author={Ma, Jiali and Yue, Zhongqi and Tomoyuki, Kagaya and Tomoki, Suzuki and Jayashree, Karlekar and Pranata, Sugiri and Zhang, Hanwang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20861--20870},
  year={2023}
}
```
## References & Opensources
Part of our implementation is adopted from the Arcface, IP-IRM, TFace and InvariantRiskMinimization repositories.
* Arcface(https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
* IP-IRM(https://github.com/Wangt-CN/IP-IRM)
* TFace(https://github.com/Tencent/TFace/tree/v0.1.0)
* InvariantRiskMinimization(https://github.com/facebookresearch/InvariantRiskMinimization)

## License
InvReg is ![MIT-licensed](https://github.com/PanasonicConnect/InvReg/blob/main/LICENSE).
