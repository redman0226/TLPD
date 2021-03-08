

DATA_PATH = 'DATASET'
datasets_map = {
##    'crowdhuman_fbox': {
##        'img_path':DATA_PATH+'/CrowdHuman/Images',
##        'json_path':DATA_PATH+'/CrowdHuman/annotation_train_coco.json',
##        'val_json_path':DATA_PATH+'CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'crowdhuman_vbox': {
##        'img_path':DATA_PATH+'/CrowdHuman/Images',
##        'json_path':DATA_PATH+'/CrowdHuman/vbox_annotations/train.json',
##        'val_json_path':DATA_PATH+'/CrowdHuman/vbox_annotations/val.json',
##       
##    },
##    'crowdhuman_hbox': {
##        'img_path':DATA_PATH+'/CrowdHuman/Images',
##        'json_path':DATA_PATH+'/CrowdHuman/head_annotations/train.json',
##        'val_json_path':DATA_PATH+'/CrowdHuman/head_annotations/val.json',
##       
##    },
    
###############
    
##    'CHnPD_fbox_fold1': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold1.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold2': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold2.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold3': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold3.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold4': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold4.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold5': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold5.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    }

##    'CHnPD_fbox_fold1': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images' , DATA_PATH+'/CrowdHuman/Images_newaug' , DATA_PATH+'/PANDA/cluster_images_newaug'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold1.json',DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold1.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold2': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images' , DATA_PATH+'/CrowdHuman/Images_newaug' , DATA_PATH+'/PANDA/cluster_images_newaug'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold2.json',DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold2.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold3': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images' , DATA_PATH+'/CrowdHuman/Images_newaug' , DATA_PATH+'/PANDA/cluster_images_newaug'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold3.json',DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold3.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold4': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images' , DATA_PATH+'/CrowdHuman/Images_newaug' , DATA_PATH+'/PANDA/cluster_images_newaug'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold4.json',DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold4.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    },
##    'CHnPD_fbox_fold5': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images' , DATA_PATH+'/CrowdHuman/Images_newaug' , DATA_PATH+'/PANDA/cluster_images_newaug'],
##        'json_path':[DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold5.json',DATA_PATH+'/CrowdHuman/annotation_train_coco.json',DATA_PATH+'/PANDA/annotations/panda_train_fold5.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
##       
##    }

    'CHnPD_fbox_fold1': {
        'img_path':[DATA_PATH+'/PANDA/cluster_images',DATA_PATH+'/PANDA/cluster_images_newaug'],
        'json_path':[DATA_PATH+'/PANDA/annotations/panda_train_fold1.json',DATA_PATH+'/PANDA/annotations/panda_train_fold1.json'],
        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
       
    },
    'CHnPD_fbox_fold2': {
        'img_path':[DATA_PATH+'/PANDA/cluster_images',DATA_PATH+'/PANDA/cluster_images_newaug'],
        'json_path':[DATA_PATH+'/PANDA/annotations/panda_train_fold2.json',DATA_PATH+'/PANDA/annotations/panda_train_fold2.json'],
        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
       
    },
    'CHnPD_fbox_fold3': {
        'img_path':[DATA_PATH+'/PANDA/cluster_images',DATA_PATH+'/PANDA/cluster_images_newaug'],
        'json_path':[DATA_PATH+'/PANDA/annotations/panda_train_fold3.json',DATA_PATH+'/PANDA/annotations/panda_train_fold3.json'],
        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
       
    },
    'CHnPD_fbox_fold4': {
        'img_path':[DATA_PATH+'/PANDA/cluster_images',DATA_PATH+'/PANDA/cluster_images_newaug'],
        'json_path':[DATA_PATH+'/PANDA/annotations/panda_train_fold4.json',DATA_PATH+'/PANDA/annotations/panda_train_fold4.json'],
        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
       
    },
    'CHnPD_fbox_fold5': {
        'img_path':[DATA_PATH+'/PANDA/cluster_images',DATA_PATH+'/PANDA/cluster_images_newaug'],
        'json_path':[DATA_PATH+'/PANDA/annotations/panda_train_fold5.json',DATA_PATH+'/PANDA/annotations/panda_train_fold5.json'],
        'val_json_path':DATA_PATH+'/CrowdHuman/annotation_val_coco.json',
       
    }

###############

##    'CHnPD_vbox': {
##        'img_path':[DATA_PATH+'/CrowdHuman/Images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/CrowdHuman/vbox_annotations/train.json',DATA_PATH+'/PANDA/annotations/panda_train.json'],
##        'val_json_path':DATA_PATH+'/CrowdHuman/vbox_annotations/val.json',
##       
##    }

##    'PD_fbox': {
##        'img_path':[DATA_PATH+'/PANDA/cluster_images',DATA_PATH+'/PANDA/cluster_images'],
##        'json_path':[DATA_PATH+'/PANDA/annotations/panda_train.json',DATA_PATH+'/PANDA/annotations/panda_test.json'],   
##        'val_json_path':DATA_PATH+'/PANDA/annotations/panda_train.json'
##
##    }

}
