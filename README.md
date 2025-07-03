# Visual-Reasoning-and-Retroduction

## 🔓 Code Availability

The related code will be released progressively.

## 🖼️Dataset and Annotation

The following is an example from EPIC dataset:

```json
[
  {
    "folder_name": "s201",
    "image_id": "201086",
    "image": "s201/frame_0086.jpg",
    "caption": "Daddy Pig makes a large bubble with his mixture.",
    "Annotation": "Daddy Pig holds a racket covered in mixture.",
    "Timestamp": "00:03:25 - 00:03:33"
  },
  {
    "folder_name": "s201",
    "image_id": "201087",
    "image": "s201/frame_0087.jpg",
    "caption": "Daddy Pig makes a large bubble with his mixture.",
    "Annotation": "Daddy Pig opens his mouth.",
    "Timestamp": "00:03:25 - 00:03:33"
  },
  {
    "folder_name": "s201",
    "image_id": "201088",
    "image": "s201/frame_0088.jpg",
    "caption": "Daddy Pig makes a large bubble with his mixture.",
    "Annotation": "Daddy Pig is blowing bubbles.",
    "Timestamp": "00:03:25 - 00:03:33"
  },
  {
    "folder_name": "s201",
    "image_id": "201089",
    "image": "s201/frame_0089.jpg",
    "caption": "Daddy Pig makes a large bubble with his mixture.",
    "Annotation": "Daddy Pig makes his own bubble.",
    "Timestamp": "00:03:25 - 00:03:33"
  },
  {
    "folder_name": "s201",
    "image_id": "201090",
    "image": "s201/frame_0090.jpg",
    "caption": "Daddy Pig had started making bubbles.",
    "Annotation": "A bubble from Daddy Pig floating in the air.",
    "Timestamp": "00:03:25 - 00:03:33"
  }
]
```

The following is an example from COIN-IC dataset:

```json
[
  {
    "folder_name": "gyjaRD5BDks",
    "image_id": "Dks001",
    "image": "gyjaRD5BDks_step1.jpg",
    "caption": "Someone replaces the screen protector on the phone.",
    "Annotation": "paste protector on the screen",
    "segment": "[20.0, 29.0]"
  },
  {
    "folder_name": "gyjaRD5BDks",
    "image_id": "Dks002",
    "image": "gyjaRD5BDks_step2.jpg",
    "caption": "Someone replaces the screen protector on the phone.",
    "Annotation": "wipe the screen",
    "segment": "[39.0, 57.0]"
  },
  {
    "folder_name": "gyjaRD5BDks",
    "image_id": "Dks003",
    "image": "gyjaRD5BDks_step3.jpg",
    "caption": "Someone replaces the screen protector on the phone.",
    "Annotation": "remove the original protector",
    "segment": "[63.0, 66.0]"
  },
  {
    "folder_name": "gyjaRD5BDks",
    "image_id": "Dks004",
    "image": "gyjaRD5BDks_step4.jpg",
    "caption": "Someone replaces the screen protector on the phone.",
    "Annotation": "line up the protector and the cellphone",
    "segment": "[71.0, 81.0]"
  },
  {
    "folder_name": "gyjaRD5BDks",
    "image_id": "Dks005",
    "image": "gyjaRD5BDks_step5.jpg",
    "caption": "Someone replaces the screen protector on the phone.",
    "Annotation": "wipe the screen",
    "segment": "[82.0, 96.0]"
  }
]
```

## Environment

This is the PyTorch code of the VRR. The code has been tested on PyTorch 1.13.1. To install the dependencies, run

```bash
conda create -n vrr python=3.7
conda activate vrr
pip install -r requirements.txt
```

## Acknowledgement

The implementation of VRR relies on resources from [BLIP](https://github.com/salesforce/BLIP), [BERTScore](https://github.com/Tiiiger/bert_score). We thank the original authors for their open-sourcing.


