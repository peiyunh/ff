# Safe Local Motion Planning with Self-Supervised Freespace Forecasting
By Peiyun Hu, Aaron Huang, John Dolan, David Held, and Deva Ramanan

## Citing us
You can find our paper on [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Safe_Local_Motion_Planning_With_Self-Supervised_Freespace_Forecasting_CVPR_2021_paper.pdf). If you find our work useful, please consider citing:
```
@inproceedings{hu2021safe,
  title={Safe Local Motion Planning with Self-Supervised Freespace Forecasting},
  author={Hu, Peiyun and Huang, Aaron and Dolan, John and Held, David and Ramanan, Deva},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12732--12741},
  year={2021}
}
```

## Training
Run the following command to start training our model. 
```
python script.py train_nuscenes --base_cfg="all.pp.mhead.vpn.config" --sample_factor=1 --epochs=20 --eval_epoch=2 --sweep_db=True --label=vp_pp_oa_ta_learn --resume=True
```

## Testing
Run the following command to start training our model. 
```
python script.py train_nuscenes --base_cfg="all.pp.mhead.vpn.config" --sample_factor=1 --epochs=20 --eval_epoch=2 --sweep_db=True --label=vp_pp_oa_ta_learn --resume=True
```

## Evaluation

