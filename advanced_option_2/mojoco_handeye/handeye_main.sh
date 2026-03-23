#!/bin/bash

echo "run foundationPose_handeye ..."
cd FoundationPose
python3 foundationPose_handeye.py

echo "run export_selected_poses_json ..."
cd ..
python3 export_selected_poses_json.py

echo "run cal_handeye ..."
python3 cal_handeye_opencv.py