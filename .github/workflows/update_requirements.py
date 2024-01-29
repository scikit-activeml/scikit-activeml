import argparse

start_version = "0.2"

def create_switcher(versions, rel_version, versions_short, rel_version_short):
    # Create an entry for every version
    content_list = []
    content_list.append("[\n")
    content_list.append("  {\n")
    content_list.append(f'    "name": "latest",\n')
    content_list.append(f'    "version": "latest",\n')
    content_list.append(f'    "url": "https://alexanderbenz.github.io/scikit-activeml-project-docs/latest/"\n')
    content_list.append("  },\n")
    for (ver, ver_s) in zip(versions, versions_short):
        content_list.append("  {\n")
        content_list.append(f'    "name": "{ver_s}",\n')
        if rel_version_short == ver_s:
            content_list.append(f'    "version": "{rel_version}",\n')
        else:
            content_list.append(f'    "version": "{ver}",\n')
        content_list.append(f'    "url": "https://alexanderbenz.github.io/scikit-activeml-project-docs/{ver_s}/"\n')
        content_list.append("  },\n")
    content_list[-1] = "  }\n"
    content_list.append("]")
    return content_list

# Get all versions an new Release version 
parser = argparse.ArgumentParser(description='')
parser.add_argument('--versions', dest='versions', nargs='+',type=str, default="0.0.0")
parser.add_argument('--release_version', dest='release_version', type=str, default="0.0.1")
args = parser.parse_args()

# Load and split versions for each release
cmd_args = vars(args)
versions = cmd_args['versions']
rel_version = cmd_args['release_version']
rel_version_split = rel_version.split(".")
rel_version_short = f"{rel_version_split[0]}.{rel_version_split[1]}"

versions_short = []
latest_versions = []
latest_version = -1
# Differential all minor from major releases
for version in versions:
    version_split = version.split(".")
    version = f"{version_split[0]}.{version_split[1]}"
    if float(version) < float(start_version):
        continue
    
    if version not in versions_short:
        latest_version = int(version_split[2])
        versions_short.append(version)
        latest_versions.append(f"{version}.{latest_version}")
    else:
        if int(version_split[2]) > latest_version:
            latest_version = int(version_split[2])
            latest_versions[-1] = f"{version}.{latest_version}"
        


content_list = create_switcher(latest_versions, rel_version, versions_short, rel_version_short)
# Override requirements.
with open("docs/_static/switcher.json", "w") as f:
    for item in content_list:
        f.write("%s" % item)