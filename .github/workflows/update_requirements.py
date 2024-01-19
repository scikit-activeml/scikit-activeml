from pip._internal.operations import freeze
# TODO: create switscher daraus erstellen. mit git tag alle durchgehen.
# versionen pro 0.* version erstellen
# anschauen wie man andere workflows canceled
# Load currently installed packages.
packages = list(freeze.freeze())
packages_dict = {}
for p in packages:
    print(p)
    try:
        p_split = p.split("==")
        version_split = p_split[1].split(".")
        version = f"{version_split[0]}.{int(version_split[1]) + 1}"
        packages_dict[p_split[0]] = version
    except:
        continue

# Read requirements and add upper bounds.
req = open("requirements.txt", "r")
content_list = req.readlines()
print(packages_dict.keys())
for line_idx, line in enumerate(content_list):
    print(line)
    try:
        name = line.split(">=")[0]
        line = line.replace("\n", "")
        line += f",<{packages_dict[name]}\n"
        content_list[line_idx] = line
    except:
        continue


# Override requirements.
with open("requirements.txt", "w") as f:
    for item in content_list:
        f.write("%s" % item)
