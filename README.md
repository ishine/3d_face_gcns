# 3d_face_gcns

TODO
1. Create 'data' directory and subdirectroy freely and put target_video in created directory. ex)  'data/studio1'
2. Create 'clip' directory in 'data/studio1' directory, and put video clips ex) 'data/studio1/clip/studio_1_0.mp4'.
3. Create 'textgrid' directory in 'data/studio1' directory, and put textgrid files ex) 'data/studio1/textgrid/studio_1_0.TextGrid'


2. In https://drive.google.com/drive/u/0/folders/11LuLtRMU-f_AVO0hOI031bXSx19g7IEe , download all files and
    - put s3fd.pth in 'audiodvp_utils/face_detection/detection/sfd' directory
    - create 'weights' directory and put 'resnet50_ft_weight.pkl'
 
3. Follow 'scripts/merge.sh' to create merged video and flist.txt file
4. Follow 'scripts/phoneme_match.sh' for lip sync generation
