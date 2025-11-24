def get_default_config():
    return {
        'train_npz': 'data/teacher_features/mediapipe_pca1024/train',
        'dev_npz': 'data/teacher_features/mediapipe_pca1024/dev',
        'test_npz': 'data/teacher_features/mediapipe_pca1024/test',
        'train_csv': 'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv',
        'dev_csv': 'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv',
        'test_csv': 'data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/test.SI5.corpus.csv',

        'input_dim': 1024,
        'hidden_dim': 512,
        'use_temporal_conv': True,

        'phase1_epochs': 25,
        'phase1_lr_start': 1e-6,
        'phase1_lr_end': 1e-3,
        'phase1_warmup_epochs': 5,
        'phase1_blank_penalty': -5.0,
        'phase1_temperature': 1.5,
        'phase1_dropout_frame': 0.1,
        'phase1_dropout_sequence': 0.1,
        'phase1_time_mask_prob': 0.0,
        'phase1_weight_decay': 1e-5,
        'phase1_max_seq_len': None,

        'phase2_epochs': 80,
        'phase2_lr': 1e-3,
        'phase2_blank_penalty_start': -6.0,
        'phase2_blank_penalty_end': -2.0,
        'phase2_dropout_frame': 0.15,
        'phase2_dropout_sequence': 0.15,
        'phase2_time_mask_prob': 0.2,
        'phase2_weight_decay': 1e-4,
        'phase2_max_seq_len': None,
        'phase2_scheduler_patience': 12,
        'phase2_scheduler_factor': 0.5,
        'phase2_temperature': 1.5,

        'phase3_epochs': 50,
        'phase3_blank_penalty': -0.5,
        'phase3_dropout_frame': 0.2,
        'phase3_dropout_sequence': 0.25,
        'phase3_time_mask_prob': 0.15,
        'phase3_weight_decay': 1e-4,
        'phase3_max_seq_len': 300,
        'phase3_scheduler_patience': 8,
        'phase3_scheduler_factor': 0.6,
        'phase3_temperature': 1.2,

        'phase4_epochs': 30,
        'phase4_blank_penalty': 0.0,
        'phase4_dropout_frame': 0.25,
        'phase4_dropout_sequence': 0.35,
        'phase4_time_mask_prob': 0.05,
        'phase4_weight_decay': 1e-4,
        'phase4_max_seq_len': 300,
        'phase4_scheduler_patience': 6,
        'phase4_scheduler_factor': 0.7,
        'phase4_temperature': 1.0,

        'batch_size': 16,
        'gradient_clip': 10.0,
        'seed': 42,
        'checkpoint_dir': 'checkpoints/hierarchical_mediapipe_pca1024',
        'log_dir': 'logs/hierarchical_mediapipe_pca1024',
        'experiment_name': 'hierarchical_mediapipe_pca1024_v1',
        'target_wer': 35.0
        ,
        # Decoding
        'decode_method': 'greedy',  # 'greedy' or 'beam_search'
        'beam_width': 10
    }


