import gdown
import zipfile

file_id = "17QRYHBBmzTmwrAWrZ-F5Qo8i7rgnP6u2"
output = "./checkpoint-500.zip"
gdown.download(id=file_id, output=output, quiet=False)

output_dir = "./checkpoint-500"
zip_file = zipfile.ZipFile(output)
zip_file.extractall(path=output_dir)
