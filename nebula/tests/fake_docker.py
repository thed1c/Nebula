#!/usr/bin/env python3
"""A stand-in for `docker`, for the tests: `volume create … --opt device=<dir> <name>` is recorded in
FAKE_DOCKER_VOLUMES (a JSON file: name -> dir) so the fake openshell can map a sandbox's mounts to folders here;
`volume inspect` knows the recorded ones; `build -t <tag> <folder>` keeps a copy of the folder and records it in
FAKE_DOCKER_IMAGES (tag -> the copy) so the fake openshell can run a program from that image, and `image inspect`
knows those; everything else succeeds and does nothing."""
import json
import os
import shutil
import sys


def main(argv):
    path = os.environ.get("FAKE_DOCKER_VOLUMES")
    vols = json.load(open(path)) if path and os.path.exists(path) else {}
    if argv[:2] == ["volume", "inspect"]:
        return 0 if argv[2] in vols else 1
    if argv[:2] == ["volume", "create"] and path:
        dev = next((a.split("=", 1)[1] for a in argv if a.startswith("device=")), None)
        vols[argv[-1]] = dev
        json.dump(vols, open(path, "w"))
    ipath = os.environ.get("FAKE_DOCKER_IMAGES")
    images = json.load(open(ipath)) if ipath and os.path.exists(ipath) else {}
    if argv[:1] == ["build"] and ipath and "-t" in argv and argv[-1] != "-":
        tag = argv[argv.index("-t") + 1]
        copy = os.path.join(ipath + ".d", tag.replace(":", "_").replace("/", "_"))
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(argv[-1], copy)
        images[tag] = copy
        json.dump(images, open(ipath, "w"))
    if argv[:2] == ["image", "inspect"]:
        return 0 if argv[2] in images else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
