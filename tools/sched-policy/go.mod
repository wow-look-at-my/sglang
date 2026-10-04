module github.com/wow-look-at-my/sglang/tools/sched-policy // go-toolchain:generate=4e3a76c3aa6e

go 1.26

require github.com/wow-look-at-my/go-ipc v0.0.0 // go-toolchain:branch=typed-service

require github.com/wow-look-at-my/go-ipc/ipcgen v0.0.0 // indirect; go-toolchain:branch=typed-service

require (
	github.com/stretchr/testify v1.12.1
	github.com/wow-look-at-my/go-containers v0.0.0
)

require (
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/spf13/cobra v1.10.2 // indirect
	github.com/spf13/pflag v1.0.9 // indirect
	github.com/wow-look-at-my/go-mmap v0.0.0 // indirect
	github.com/wow-look-at-my/go-shm v0.0.0 // indirect
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.33.0 // indirect
)

tool github.com/wow-look-at-my/go-ipc/ipcgen
