#python main.py --task_type=css --dae_model_type=DeepStack --dae_model_path=save_model/DAE_model/2022-08-17/1400_dae_checkpoint.pth --meta_model_path=save_model/MLP_model/2022-08-18/60_2022-08-18_01:37_mlp_checkpoint.pth  --meta_model_type=MLP --train_dae_model=0 --train_meta_model=0
#python main.py --task_type=css --dae_model_type=DeepStack --dae_model_path=save_model/DAE_model/2022-08-17/1400_dae_checkpoint.pth  --meta_model_type=Catboost --train_dae_model=0 --train_meta_model=1

#python main.py --task_type=css --dae_model_type=DeepStack --dae_model_path=save_model/DAE_model/2022-08-17/1400_dae_checkpoint.pth  --meta_model_type=MLP --train_dae_model=0 --train_meta_model=1
#python main.py --task_type=css --dae_model_type=DeepStack --meta_model_type=MLP --dae_model_path=save_model/DAE_model/2022-08-18/1000_16:19_dae_checkpoint.pth --train_dae_model=0 --train_meta_model=1
#python main.py --task_type=css --dae_model_type=DeepStack --meta_model_type=Catboost --dae_model_path=save_model/DAE_model/2022-08-18/1000_16:19_dae_checkpoint.pth --train_dae_model=0 --train_meta_model=1
#python main.py --task_type=css --dae_model_type=DeepStack --meta_model_type=Catboost --dae_model_path=save_model/DAE_model/2022-08-18/1000_16:19_dae_checkpoint.pth --train_dae_model=0 --train_meta_model=1

#python main.py --task_type=css --train_data_path='data/X_cds_train.parquet' --test_data_path='data/X_cds_test.parquet' --dae_model_type=DeepStack --meta_model_type=MLP --train_dae_model=1 --train_meta_model=1
#python main.py --task_type=css --train_data_path='../../CSS/dataset/X_dec_qt_scaled.parquet' --dae_model_type=DeepBottleneck --train_dae_model=0 --dae_model_path='save_model/DAE_model/2022-09-25/1100_02:42_dae_checkpoint.pth' --train_meta_model=1 --meta_model_type=Catboost
python main.py --task_type=css --train_data_path='../../CSS/dataset/X_jan_qt_scaled.parquet' --dae_model_type=DeepBottleneck --train_dae_model=0 --dae_model_path='save_model/DAE_model/2022-09-25/1100_02:42_dae_checkpoint.pth' --train_meta_model=0 --meta_model_type=MLP --meta_model_path='save_model/MLP_model/2022-09-26/60_11:36_mlp_checkpoint.pth'