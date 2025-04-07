nohup env CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node 4 --master_port 8255 train.py --data data/aitod.yaml \
 --img 800 --device 3,4,5,6 --cfg models/yolov5s.yaml --patience 300 \
 --batch-size 128 --project aitod_yolos/5s_nopretrain_ep500_300stop --epochs 1000 > logs/5s_nopretrain_ep500_300stop.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 9225 train.py --data data/aitod.yaml \
 --img 800 --device 0,1,2,3 --cfg models/yolov5s_neck_c3faster_ema.yaml \
--batch-size 32 --project aitod_yolos/faster_ema_nopretrain --name neck --epochs 200 > logs/yolov5s_c3faster_ema_neck.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node 1 --master_port 8225 train.py --data data/aitod.yaml \
 --img 800 --device 7 --cfg models/yolov5s_c3faster_ema.yaml \
--batch-size 32 --project aitod_yolos/faster_ema_nopretrain --epochs 200 > logs/yolov5s_c3faster_ema.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 8221 train.py --data data/aitod.yaml \
 --img 800 --device 2,5,6,7 --cfg models/yolov5s_DBB.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_DBB --epochs 200 > logs/yolov5s_DBB.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.run --nproc_per_node 1 --master_port 8221 train.py --data data/aitod.yaml \
 --img 800 --device 5 --cfg models/yolov5s_SPPF_LSKA.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_SPPF_LSKA --epochs 200 > logs/yolov5s_SPPF_LSKA.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port 4521 train.py --data data/aitod.yaml \
 --img 800 --device 6,7 --cfg models/yolov5s_fasterHead.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_fasterHead --epochs 200 > logs/yolov5s_fasterHead.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node 2 --master_port 4521 train.py --data data/aitod.yaml \
 --img 800 --device 3,4 --cfg models/yolov5s_ScConv.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_ScConv --epochs 200 > logs/yolov5s_ScConv.log 2>&1 &

#--------------------------------------------------------------yolov5_p2----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node 2 --master_port 4522 train.py --data data/aitod.yaml \
 --img 800 --device 3,4 --cfg models/yolov5_p2.yaml \
--batch-size 32 --project aitod_yolos/yolov5_p2 --epochs 200 > logs/yolov5_p2.log 2>&1 &
#--------------------------------------------------------------yolov5_p2----------------------------------------------------------------------------#

#--------------------------------------------------------------yolov5s_p2_BH----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 4522 train.py --data data/aitod.yaml \
 --img 800 --device 2,3 --cfg models/yolov5s_p2_BH.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_BH --epochs 200 > logs/yolov5s_p2_BH.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_BH----------------------------------------------------------------------------#

#--------------------------------------------------------------yolov5s_ema_fpn----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 14522 train.py --data data/aitod.yaml \
 --img 800 --device 2,3 --cfg models/yolov5s_ema_fpn.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_ema_fpn --epochs 200 > logs/yolov5s_ema_fpn.log 2>&1 &
#--------------------------------------------------------------yolov5s_Biformer_P2----------------------------------------------------------------------------#

#--------------------------------------------------------------yolov5s_p2_ema----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 14322 train.py --data data/aitod.yaml \
 --img 800 --device 2,3 --cfg models/yolov5s_p2_ema.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_ema --epochs 200 > logs/yolov5s_p2_ema.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_ema----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_DLKA----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 14322 train.py --data data/aitod.yaml \
 --img 800 --device 2,3 --cfg models/yolov5s_DLKA.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_DLKA --epochs 200 > logs/yolov5s_DLKA.log 2>&1 &
#--------------------------------------------------------------yolov5s_DLKA----------------------------------------------------------------------------#

#--------------------------------------------------------------yolov5s_p2_DLKA----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 14321 train.py --data data/aitod.yaml \
 --img 800 --device 2,3 --cfg models/yolov5s_p2_DLKA.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_DLKA --epochs 200 > logs/yolov5s_p2_DLKA.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_DLKA----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_p2_rep----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=4,7 python -m torch.distributed.run --nproc_per_node 2 --master_port 14324 train.py --data data/aitod.yaml \
 --img 800 --device 4,7 --cfg models/yolov5s_p2_rep.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_rep --epochs 200 > logs/yolov5s_p2_rep.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_rep----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_p2_rep_27rep----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=4,7 python -m torch.distributed.run --nproc_per_node 2 --master_port 14324 train.py --data data/aitod.yaml \
 --img 800 --device 4,7 --cfg models/yolov5s_p2_rep_27rep.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_rep_27rep --epochs 200 > logs/yolov5s_p2_rep_27rep.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_rep_27rep----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_p2345----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 14321 train.py --data data/aitod.yaml \
 --img 800 --device 2,3 --cfg models/yolov5s_p2345.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2345 --epochs 200 > logs/yolov5s_p2345.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2345----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_p2_BH_repema----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node 2 --master_port 14321 train.py --data data/aitod.yaml \
 --img 800 --device 3,4 --cfg models/yolov5s_p2_BH_repema.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_BH_repema --epochs 200 > logs/yolov5s_p2_BH_repema.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_BH_repema----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_p2_BH_repema_311----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node 2 --master_port 11321 train.py --data data/aitod.yaml \
 --img 800 --device 5,6 --cfg models/yolov5s_p2_BH_repema_311.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p2_BH_repema_311 --epochs 200 > logs/yolov5s_p2_BH_repema_311.log 2>&1 &
#--------------------------------------------------------------yolov5s_p2_BH_repema_311----------------------------------------------------------------------------#

#--------------------------------------------------------------yolov5s_RCSOSA----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node 2 --master_port 11321 train.py --data data/aitod.yaml \
 --img 800 --device 4,5 --cfg models/yolov5s_RCSOSA.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_RCSOSA --epochs 200 > logs/yolov5s_RCSOSA.log 2>&1 &
#--------------------------------------------------------------yolov5s_RCSOSA----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_p23----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node 2 --master_port 11321 train.py --data data/aitod.yaml \
 --img 800 --device 3,4 --cfg models/yolov5s_p23.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_p23 --epochs 200 > logs/yolov5s_p23.log 2>&1 &
#--------------------------------------------------------------yolov5s_p23----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_nopretrain_ep300----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 11322 train.py --data data/aitod.yaml \
 --img 800 --device 0,1,2,3 --cfg models/yolov5s.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_nopretrain_ep300 --epochs 300 > logs/yolov5s_nopretrain_ep300.log 2>&1 &
#--------------------------------------------------------------yolov5s_nopretrain_ep300----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5s_BH_c3faster_ema_ep300----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=1,2,3,5 python -m torch.distributed.run --nproc_per_node 4 --master_port 12222 train.py --data data/aitod.yaml \
 --img 800 --device 1,2,3,5 --cfg models/yolov5s_BH_c3faster_ema.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_BH_c3faster_ema_ep300 --epochs 300 > logs/yolov5s_BH_c3faster_ema_ep300.log 2>&1 &
#--------------------------------------------------------------yolov5s_BH_c3faster_ema_ep300----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5m----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --nproc_per_node 2 --master_port 12222 train.py --data data/aitod.yaml \
 --img 800 --device 1,2 --cfg models/yolov5m.yaml \
--batch-size 32 --project aitod_yolos/yolov5m --epochs 200 > logs/yolov5m.log 2>&1 &
#--------------------------------------------------------------yolov5m----------------------------------------------------------------------------#





#--------------------------------------------------------------yolov5s_faketrain_realtest----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run --nproc_per_node 1 --master_port 12233 train.py --data data/aitod_AI_test.yaml --weight yolov5s.pt \
 --img 800 --device 3 --cfg models/yolov5s.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_faketrain_realtest --epochs 200 > logs/yolov5s_faketrain_realtest.log 2>&1 &
#--------------------------------------------------------------yolov5s_faketrain_realtest----------------------------------------------------------------------------#



nohup env CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --nproc_per_node 2 --master_port 12221 train.py --data data/aitod.yaml \
 --img 800 --device 3,4 --cfg models/yolov5s_LAWDS.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_LAWDS --epochs 200 > logs/yolov5s_LAWDS.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.run --nproc_per_node 2 --master_port 12321 train.py --data data/aitod.yaml \
 --img 800 --device 5,6 --cfg models/yolov5s_dyhead_dcnv3.yaml \
--batch-size 32 --project aitod_yolos/yolov5s_dyhead_dcnv3 --epochs 200 > logs/yolov5s_dyhead_dcnv3.log 2>&1 &

#--------------------------------------------------------------yolov5m_fake_tune_real----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 12321 train.py --data data/aitod_AI_test.yaml --weights /data/home/thebs/yolos/yolov5_Biformer/plane15m/midmodel7/weights/best.pt \
 --img 1024 --device 4,5,6,7 --cfg /data/home/thebs/yolos/yolov5_Biformer/models/yolo5s_Biformer_BLM_FPN.yaml \
--batch-size 16 --project aitod_yolos/yolov5m_fake_tune_real --epochs 100 > logs/yolov5m_fake_tune_real.log 2>&1 &
#--------------------------------------------------------------yolov5m_fake_tune_real----------------------------------------------------------------------------#


#--------------------------------------------------------------yolov5m_real----------------------------------------------------------------------------#
nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port 12322 train.py --data data/aitod_AI_test.yaml --weights yolov5s.pt \
 --img 1024 --device 4,5,6,7 --cfg models/yolov5s.yaml \
--batch-size 16 --project aitod_yolos/yolov5m_real --epochs 100 > logs/yolov5m_real.log 2>&1 &
#--------------------------------------------------------------yolov5m_real----------------------------------------------------------------------------#





nohup env CUDA_VISIBLE_DEVICES=2 python train.py --data data/aitod.yaml \
--weights yolov5s.pt --img 800 --device 2 --cfg models/yolov5s.yaml \
--batch-size 32 --project aitod_yolos --epochs 300 > logs/aitod_yolos_exp1.log 2>&1 &



############################################################VAL#########################################################################################
python val.py --data data/aitod.yaml --weights aitod_yolos/yolov5m/exp2/weights/best.pt --img 800 --task test --device 2 --verbose --batch-size 1
########################################################################################################################################################




CUDA_VISIBLE_DEVICES=2 python detect.py --weights aitod_yolos/yolov5s_p2_BH_v6head/exp/weights/best.pt --source /data/home/thebs/mydataset/AI-TOD/dataset/train/images --img 800 --device 7 --nosave