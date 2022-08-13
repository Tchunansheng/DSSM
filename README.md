## [Doubly Stochastic Subdomain Mining with Sample Reweighting for Unsupervised Domain Adaptive Person Re-identification]


### Preparation

#### Requirements: Python=3.6 and Pytorch>=1.0.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset

   - Market-1501  [[GoogleDriver]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?usp=sharing) 
   
   - DukeMTMC-reID [[GoogleDriver]](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O)     
   - Unzip each dataset   
   Ensure the File structure is as follow:
   
   ```
   dataset_path    
   │
   └───market OR dukemtmc
      │   
      └───bounding_box_train
      │   
      └───bounding_box_test
      │   
      └───query
   ```

### Training and test DSSM model for person re-ID

  ```Shell
  # For Market to Duke
  python examples/train.py --dataset_source market --dataset_target dukemtmc --eps 0.72 --logs_dir logs/market2duke

  # For Duke to Market
  python examples/train.py --dataset_source dukemtmc --dataset_target market --eps 0.56 --logs_dir logs/duke2market

  ```
    
### Contact me

If you have any questions about this code, please do not hesitate to contact me.

email: 1910316@stu.neu.edu.cn

