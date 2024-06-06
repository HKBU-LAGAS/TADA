# photo
# done
python main.py --model GCN --dataset photo  --cfg ./cfg/gcn.yaml --gpu 0  
python main.py --model GCN2 --dataset photo --cfg ./cfg/gcn2.yaml --gpu 0
python main.py --model EbdGNN --dataset photo   --cfg ./cfg/TADA_gcn.yaml  --se_type 'rwr' --se_dim 256 --gamma 0.3 --rho 0.3 --beta 1.0  --gpu 0 
python main.py --model EbdGNN --dataset photo --cfg ./cfg/TADA_gcn2.yaml  --se_type 'rwr' --se_dim 256  --gamma 0.3 --rho 0.3 --beta 1.0  --gpu 0

# squirrel
# done 
python main.py --model GCN --dataset squirrel  --cfg ./cfg/gcn.yaml --gpu 0
python main.py --model GCN2 --dataset squirrel --cfg ./cfg/gcn2.yaml --gpu 0
python main.py --model EbdGNN --dataset squirrel  --cfg ./cfg/TADA_gcn.yaml  --se_type 'rwr' --se_dim 128 --gamma 1.0 --rho 0.5 --beta 1.0  --gpu 0
python main.py --model EbdGNN --dataset squirrel --cfg ./cfg/TADA_gcn2.yaml  --se_type 'rwr' --se_dim 128 --gamma 1.0 --rho 0.5 --beta 1.0  --gpu 0

# wikics
python main.py --model GCN --dataset wikics  --cfg ./cfg/gcn.yaml --gpu 0
python main.py --model GCN2 --dataset wikics --cfg ./cfg/gcn2.yaml --gpu 0
python main.py --model EbdGNN --dataset wikics  --cfg ./cfg/TADA_gcn.yaml  --se_type 'rwr' --se_dim 128 --gamma 0.3 --rho 0.1  --beta 1.0  --gpu 0 
python main.py --model EbdGNN --dataset wikics --cfg ./cfg/TADA_gcn2.yaml  --se_type 'rwr' --se_dim 128 --gamma 0.2 --rho 0.2  --beta 1.0  --gpu 0 


# reddit
python main.py --model GCN --dataset reddit2  --cfg ./cfg/gcn.yaml --gpu 0
python main.py --model GCN2 --dataset reddit2 --cfg ./cfg/gcn2.yaml --gpu 0
python main.py --model EbdGNN --dataset reddit2  --cfg ./cfg/TADA_gcn.yaml  --se_type 'rwr' --se_dim 128 --gamma 0.5 --rho 0.7 --beta 1.0  --gpu 0
python main.py --model EbdGNN --dataset reddit2 --cfg ./cfg/TADA_gcn2.yaml  --se_type 'rwr' --se_dim 128 --gamma 0.5 --rho 0.7 --beta 1.0  --gpu 0

# proteins
python main.py --model GCN --dataset proteins  --cfg ./cfg/gcn.yaml  --gpu 0
python main.py --model GCN2 --dataset proteins --cfg ./cfg/gcn2.yaml --gpu 0
python main.py --model EbdGNN --dataset proteins  --cfg ./cfg/TADA_gcn.yaml  --se_type 'rwr' --se_dim 64  --gamma 0.5 --rho 0.9 --beta 1.0  --gpu 0
python main.py --model EbdGNN --dataset proteins --cfg ./cfg/TADA_gcn2.yaml  --se_type 'rwr' --se_dim 128 --gamma 0.5 --rho 0.9 --beta 1.0  --gpu 0