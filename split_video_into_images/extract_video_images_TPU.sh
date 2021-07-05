VIDEO_DIR=${TPU_CODE_DIR}/sample_video/

FRAME_SKIP=25

for video_num in extra_cut_1686;

do

    BASE_VIDEO_DIR=${VIDEO_DIR}

    FRAME_DIR=${BASE_VIDEO_DIR}/${video_num}_frames/
    rm -rf ${FRAME_DIR}
    mkdir -p ${FRAME_DIR}

    python cut_video_into_images.py --video-num $video_num --frame-skip $FRAME_SKIP --base-video-dir $BASE_VIDEO_DIR

done

