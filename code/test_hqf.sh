python test.py  \
--name test \
--model  eSL-Net++_sr  \
--load_G '../pre_trained/esl_plusplus_sr_pretrained.pt' \
--dataset_mode gopro \
--img_start_idx 14 \
--img_inter_idx 7 \
--num_frames_for_blur 1 \
--input_blur_path  '../test_data/HQF_test/blur_images/' \
--input_event_path '../test_data/HQF_test/eventstream_mat/' \
--output_dir '../method_results/HQF_test/' \
--cuda
