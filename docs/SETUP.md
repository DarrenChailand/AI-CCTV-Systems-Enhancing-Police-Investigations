# Setup notes

## Database

The code expects a MySQL database named `database_investigator`. Based on the current queries, the schema must provide at least these tables and fields:

- `target_pribadi`: target identifier, target name, and a Base64-encoded reference image;
- `detail_target_pribadi`: start time, end time, longitude, latitude, expression summary, and target identifier;
- `target_koneksi`: connection identifier and a Base64-encoded face image;
- `detail_target_koneksi`: target, detection-detail, and connection identifiers.

The exact original schema was not committed. Add the exported schema—without personal records or credentials—to make the project reproducible.

## Model checkpoints

`models/checkpoints/train_1.pt` through `train_4.pt` are included. The live pipeline currently loads `train_4.pt`. Document what each checkpoint represents, including its model architecture, training data, number of epochs, and validation result.

## Local paths

The original research code contains paths from a local macOS and XAMPP installation. These must be replaced with paths appropriate for the machine running the prototype. Never commit database passwords, private face images, raw investigation footage, or identifying database exports.
