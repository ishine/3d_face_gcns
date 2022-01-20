# 3d_face_gcns

TODO
1. Create 'data' directory and subdirectory for target video ex)  'data/studio1'.
2. Create subdirectory for source video(or audio) and textgrid in 'data' directory ex) 'data/studio_2_0_test'
3. Create 'clip' directory in 'data/studio1' directory, and put video clips ex) 'data/studio1/clip/studio_1_0.mp4'.
4. Create 'textgrid' directory in 'data/studio1' directory, and put textgrid files ex) 'data/studio1/textgrid/studio_1_0.TextGrid'
5. Put test video(or audio) and textgrid file in subdirectory created in step 2. ex) 'data/studio_2_0_test/studio_2_0.mp4', 'data/studio_2_0_test/studio_2_0.TextGrid'

6. In https://drive.google.com/drive/u/0/folders/11LuLtRMU-f_AVO0hOI031bXSx19g7IEe , download all files and
    - put s3fd.pth in 'audiodvp_utils/face_detection/detection/sfd' directory
    - create 'weights' directory and put 'resnet50_ft_weight.pkl'
 
7. Follow 'scripts/merge.sh' to create merged video and flist.txt file
8. Follow 'scripts/phoneme_match.sh' for lip sync generation
