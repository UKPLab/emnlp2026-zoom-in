#!/usr/bin/env python3
"""
cleanup_checkpoints.py

Goes through all *immediate* subdirectories of a parent directory (experiment dirs)
and can perform these independently toggleable actions:

1) delete-checkpoint (delete-empty-exp):
   If an experiment dir has NO subdir starting with "checkpoint-":
       delete the experiment dir

2) reduce:
   If an experiment dir has multiple "checkpoint-*" subdirs AND one starts with "checkpoint-382":
       For all other checkpoint dirs (not the 382 one):
           - if checkpoint dir contains subdir "eval":
                 delete model-0000X-of-00004.safetensors for X=1..4 inside that checkpoint dir
           - else:
                 delete the whole checkpoint dir

3) move:
   Create ONE SSH connection (username/password) and recursively copy experiment dirs to a remote base dir.

4) delete-tool-calls:
   Create ONE SSH connection (username/password).
   For each experiment dir:
       if len(local_dir/tool_calls) == len(remote_dir/tool_calls):
           delete local_dir/tool_calls

Dry-run mode prints what would happen without modifying anything.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import stat
import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Iterable, Optional, Tuple

# Optional dependency for remote ops:
#   pip install paramiko
try:
    import paramiko
except Exception:
    paramiko = None


CHECKPOINT_PREFIX = "checkpoint-"
KEEP_CHECKPOINT_PREFIX = "checkpoint-382"
TENSOR_FILES_TO_DELETE = [f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)]


@dataclass
class Options:
    parent: Path
    dry_run: bool

    enable_delete_empty_exp: bool
    enable_reduce: bool

    enable_move: bool
    remote_host: Optional[str]
    remote_port: int
    remote_user: Optional[str]
    remote_password: Optional[str]
    remote_base_dir: Optional[str]

    enable_delete_tool_calls: bool


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def safe_is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:
        return False


def list_immediate_subdirs(parent: Path) -> list[Path]:
    if not safe_is_dir(parent):
        raise ValueError(f"Parent is not a directory: {parent}")
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def checkpoint_subdirs(exp_dir: Path) -> list[Path]:
    return sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith(CHECKPOINT_PREFIX)])

def dir_is_minimal(exp_dir: Path) -> bool:
    dir_is_minimal = True
    file_list = ["log.log", "model_args.json", "run_log.txt", "script_args.json", "training_args.json", "vllm_log.txt"]
    for p in exp_dir.iterdir():
        if p.is_dir():
            if p.name != "tool_calls":
                dir_is_minimal = False
                break
        else:
            if p.name not in file_list:
                dir_is_minimal = False
                break
            else:
                file_list.remove(p.name)
    return dir_is_minimal


def has_eval_subdir(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "eval").is_dir()


def do_print(opts: Options, msg: str) -> None:
    prefix = "[DRY-RUN] " if opts.dry_run else ""
    print(prefix + msg)


def remove_tree(opts: Options, path: Path) -> None:
    if not path.exists():
        return
    do_print(opts, f"DELETE DIR  {path}")
    if not opts.dry_run:
        shutil.rmtree(path)


def remove_file(opts: Options, path: Path) -> None:
    if not path.exists():
        return
    do_print(opts, f"DELETE FILE {path}")
    if not opts.dry_run:
        path.unlink()


def assert_parent_sanity(parent: Path) -> None:
    # Basic guardrails to avoid catastrophic mistakes.
    parent = parent.resolve()
    if str(parent) in ("/", "\\"):
        raise ValueError("Refusing to operate on filesystem root.")
    if len(parent.parts) < 2:
        raise ValueError(f"Refusing to operate on suspiciously short path: {parent}")


# ---------------------------
# SSH / SFTP helpers
# ---------------------------

class RemoteOps:
    def __init__(self, host: str, port: int, user: str, password: str):
        if paramiko is None:
            raise RuntimeError("paramiko is required for remote ops. Install with: pip install paramiko")

        self.host = host
        self.port = port
        self.user = user

        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(hostname=host, port=port, username=user, password=password, timeout=20)
        self._sftp = self._ssh.open_sftp()

    def close(self) -> None:
        try:
            self._sftp.close()
        finally:
            self._ssh.close()

    def exists_dir(self, remote_path: str) -> bool:
        try:
            st = self._sftp.stat(remote_path)
            return stat.S_ISDIR(st.st_mode)
        except FileNotFoundError:
            return False

    def mkdirs(self, remote_path: str) -> None:
        # Create remote directories recursively (POSIX paths).
        parts = []
        p = remote_path
        while p not in ("", "/", None):
            parts.append(p)
            p = posixpath.dirname(p)
            if p == remote_path:
                break
        for d in reversed(parts):
            try:
                self._sftp.stat(d)
            except FileNotFoundError:
                self._sftp.mkdir(d)

    def listdir(self, remote_path: str) -> list[str]:
        return self._sftp.listdir(remote_path)

    def count_entries(self, remote_path: str) -> int:
        return len(self.listdir(remote_path))

    def put_file(self, local_path: Path, remote_path: str) -> None:
        self.mkdirs(posixpath.dirname(remote_path))
        self._sftp.put(str(local_path), remote_path)

    def _print_existing_remote_subdirs_for_local_tree(self, local_dir: Path, remote_dir: str) -> None:
        """
        Dry-run helper: for the directory tree rooted at local_dir, print which corresponding
        remote subdirectories already exist under remote_dir. Does NOT create anything.
        """
        existing: set[str] = set()

        # Only report dirs that already exist on the remote side.
        if self.exists_dir(remote_dir):
            existing.add(remote_dir)

        for root, dirs, _files in os.walk(local_dir):
            root_path = Path(root)
            rel = root_path.relative_to(local_dir)
            remote_root = remote_dir if str(rel) == "." else posixpath.join(remote_dir, *rel.parts)

            # os.walk gives us subdir names in `dirs`; check each one.
            for d in dirs:
                rp = posixpath.join(remote_root, d)
                if self.exists_dir(rp):
                    existing.add(rp)

        if existing:
            print("Existing remote subdirs (dry-run):")
            for p in sorted(existing):
                print(f"  {p}")
        else:
            print("No existing remote subdirs found (dry-run).")

    def put_dir_recursive(self, local_dir: Path, remote_dir: str, *, skip_existing: bool = False, dry_run: bool = True) -> None:
        # Simple recursive copy (no rsync semantics).

        if dry_run:
            self._print_existing_remote_subdirs_for_local_tree(local_dir, remote_dir)
            return

        self.mkdirs(remote_dir)

        for root, dirs, files in os.walk(local_dir):
            root_path = Path(root)
            rel = root_path.relative_to(local_dir)
            remote_root = remote_dir if str(rel) == "." else posixpath.join(remote_dir, *rel.parts)

            self.mkdirs(remote_root)

            for d in dirs:
                self.mkdirs(posixpath.join(remote_root, d))

            for f in files:
                lp = root_path / f
                rp = posixpath.join(remote_root, f)
                try:
                    self._sftp.stat(rp)
                    print(f"Found existing file {rp}")
                    if skip_existing:
                        continue
                except FileNotFoundError:
                    pass
                self.put_file(lp, rp)


# ---------------------------
# Actions
# ---------------------------

def action_delete_empty_exp(opts: Options, exp_dir: Path) -> None:
    chkp = checkpoint_subdirs(exp_dir)
    has_eval = has_eval_subdir(exp_dir)
    if dir_is_minimal(exp_dir) and len(chkp) == 0 and not has_eval:
        remove_tree(opts, exp_dir)


def action_reduce(opts: Options, exp_dir: Path) -> None:
    chkp = checkpoint_subdirs(exp_dir)
    if len(chkp) <= 1:
        return

    has_keep = any(p.name.startswith(KEEP_CHECKPOINT_PREFIX) for p in chkp)
    if not has_keep:
        return

    for cdir in chkp:
        if cdir.name.startswith(KEEP_CHECKPOINT_PREFIX):
            continue

        if has_eval_subdir(cdir):
            for fname in TENSOR_FILES_TO_DELETE:
                remove_file(opts, cdir / fname)
        else:
            remove_tree(opts, cdir)


def action_move(opts: Options, remote: RemoteOps, exp_dir: Path) -> None:
    assert opts.remote_base_dir is not None
    remote_exp_dir = posixpath.join(opts.remote_base_dir, exp_dir.name)
    do_print(opts, f"COPY DIR   {exp_dir}  ->  {opts.remote_user}@{opts.remote_host}:{remote_exp_dir}")
    remote.put_dir_recursive(exp_dir, remote_exp_dir, dry_run=opts.dry_run)


def count_local_entries(path: Path) -> int:
    if not path.exists():
        return 0
    if not path.is_dir():
        return 0
    try:
        return len(list(path.iterdir()))
    except OSError:
        return 0


def action_delete_tool_calls(opts: Options, remote: RemoteOps, exp_dir: Path) -> None:
    assert opts.remote_base_dir is not None, "remote_base_dir must be set for delete_tool_calls action"

    local_tc = exp_dir / "tool_calls"
    if not local_tc.is_dir():
        print(f"tool_calls dir not found for {exp_dir}")
        return

    remote_exp_dir = posixpath.join(opts.remote_base_dir, exp_dir.name)
    remote_tc = posixpath.join(remote_exp_dir, "tool_calls")

    # If remote doesn't have tool_calls dir, we do nothing (safer).
    if not remote.exists_dir(remote_tc):
        print(f"Remote does not have tool_calls dir {remote_tc}")
        return

    local_n = count_local_entries(local_tc)
    try:
        remote_n = remote.count_entries(remote_tc)
    except FileNotFoundError:
        return

    if local_n == remote_n:
        remove_tree(opts, local_tc)
    else:
        do_print(opts, f"KEEP DIR   {local_tc} (local {local_n} != remote {remote_n})")


# ---------------------------
# CLI / Main
# ---------------------------

def parse_args() -> Options:
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True, help="Parent directory containing experiment dirs")
    p.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")

    p.add_argument("--delete-empty-exp", action="store_true",
                   help='Delete experiment dir if it has no subdir starting with "checkpoint-"')
    p.add_argument("--reduce", action="store_true",
                   help='Reduce checkpoints when "checkpoint-382*" exists among multiple checkpoints')

    p.add_argument("--move", action="store_true", help="Copy experiment dirs to remote via SSH/SFTP")
    p.add_argument("--delete-tool-calls", action="store_true",
                   help="Delete local tool_calls if same entry count exists on remote")

    p.add_argument("--remote-host", default="slurm.ukp.informatik.tu-darmstadt.de", help="Remote host for SSH")
    p.add_argument("--remote-port", type=int, default=22, help="Remote SSH port (default 22)")
    p.add_argument("--remote-user", default="helm",help="Remote username for SSH")
    p.add_argument("--remote-password", default="1980EintrachtFrankfurtOle\\", help="Remote password for SSH")
    p.add_argument("--remote-base-dir", default="/mnt/beegfs/work/helm/42_data_dump/focusreason/runs", help="Remote base directory to copy into / compare against")

    args = p.parse_args()

    return Options(
        parent=Path(args.parent),
        dry_run=args.dry_run,

        enable_delete_empty_exp=args.delete_empty_exp,
        enable_reduce=args.reduce,

        enable_move=args.move,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_user=args.remote_user,
        remote_password=args.remote_password,
        remote_base_dir=args.remote_base_dir,

        enable_delete_tool_calls=args.delete_tool_calls,
    )


def need_remote(opts: Options) -> bool:
    return opts.enable_move or opts.enable_delete_tool_calls


def validate_remote_args(opts: Options) -> None:
    if not need_remote(opts):
        return
    missing = [name for name, val in [
        ("--remote-host", opts.remote_host),
        ("--remote-user", opts.remote_user),
        ("--remote-base-dir", opts.remote_base_dir),
    ] if not val]
    if missing:
        raise ValueError(f"Missing remote options required for remote actions: {', '.join(missing)}")


def main() -> int:
    opts = parse_args()
    assert_parent_sanity(opts.parent)
    validate_remote_args(opts)



    remote: Optional[RemoteOps] = None
    try:
        if need_remote(opts):
            password = opts.remote_password
            #password = getpass(f"SSH password for {opts.remote_user}@{opts.remote_host}: ")
            remote = RemoteOps(opts.remote_host, opts.remote_port, opts.remote_user, password)

        if opts.enable_delete_empty_exp:
            exp_dirs = list_immediate_subdirs(opts.parent)
            do_print(opts, f"Found {len(exp_dirs)} experiment dirs under {opts.parent}")
            for exp_dir in exp_dirs:
            # 1) delete-empty-exp (deletes whole exp dir) – do this first to avoid doing work on dirs to be deleted

                # If we delete it, skip further actions for that exp_dir
                chkp = checkpoint_subdirs(exp_dir)
                if len(chkp) == 0:
                    action_delete_empty_exp(opts, exp_dir)


        # 2) reduce
        if opts.enable_reduce:
            exp_dirs = list_immediate_subdirs(opts.parent)
            do_print(opts, f"Found {len(exp_dirs)} experiment dirs under {opts.parent}")
            for exp_dir in exp_dirs:
                action_reduce(opts, exp_dir)


        # 3) move
        if opts.enable_move and remote is not None:
            exp_dirs = list_immediate_subdirs(opts.parent)
            do_print(opts, f"Found {len(exp_dirs)} experiment dirs under {opts.parent}")
            for exp_dir in exp_dirs:
                action_move(opts, remote, exp_dir)

        # 4) delete-tool-calls (after move is often what people want)
        if opts.enable_delete_tool_calls and remote is not None:
            exp_dirs = list_immediate_subdirs(opts.parent)
            do_print(opts, f"Found {len(exp_dirs)} experiment dirs under {opts.parent}")
            for exp_dir in exp_dirs:
                print(f"start with {exp_dir}")
                action_delete_tool_calls(opts, remote, exp_dir)

        return 0
    finally:
        if remote is not None:
            remote.close()


if __name__ == "__main__":
    raise SystemExit(main())