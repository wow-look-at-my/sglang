package schedpolicy

//go:generate go tool ipcgen --lang go --out messages.go ../../schema/policy.ipc
//go:generate go tool ipcgen --lang py --out ../../../../python/sglang/srt/managers/scheduler_components/sched_policy_messages.py ../../schema/policy.ipc
