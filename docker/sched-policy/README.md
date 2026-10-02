# sched-policy binary drop

Fork CI builds `tools/sched-policy` with go-toolchain and places the APE
`sglang-sched-policy` here before the image build. The Dockerfile copies this
directory to `/opt/sglang/sched-policy` and points `SGLANG_SCHED_POLICY_BIN` at
the binary.
