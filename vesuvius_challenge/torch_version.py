import torch 
import sys 

pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
link = 'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html'

print(version_str)
print(link.format(version_str=version_str))