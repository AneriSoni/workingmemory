// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/kit"
)

// Actions are SIR actions
type Actions int

//go:generate stringer -type=Actions

var KiT_Actions = kit.Enums.AddEnum(ActionsN, kit.NotBitFlag, nil)

func (ev Actions) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Actions) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	Store1 Actions = iota
	Store2
	Store3
	Store4
	Ignore
	Recall1
	Recall2
	Recall3
	Recall4
	ActionsN
)

// SIREnv implements the store-ignore-recall task
type SIREnv struct {
	Nm        string          `desc:"name of this environment"`
	Dsc       string          `desc:"description of this environment"`
	NStim     int             `desc:"number of different stimuli that can be maintained"`
	RewVal    float32         `desc:"value for reward, based on whether model output = target"`
	NoRewVal  float32         `desc:"value for non-reward"`
	Act       Actions         `desc:"current action"`
	Stim      float64         `desc:"current stimulus"`
	Maint1    float64         `desc:"current stimulus being maintained"`
	Maint2    float64         `desc:"current stimulus being maintained"`
	Maint3    float64         `desc:"current stimulus being maintained"`
	Maint4    float64         `desc:"current stimulus being maintained"`
	Input     etensor.Float64 `desc:"stimulus input pattern"`
	CtrlInput etensor.Float64 `desc:"input pattern with action"`
	Output    etensor.Float64 `desc:"output pattern of what to respond"`
	Reward    etensor.Float64 `desc:"reward value"`
	Run       env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch     env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial     env.Ctr         `view:"inline" desc:"trial is the step counter within epoch"`
	StimType  string          `desc:"continuous stimulus or fixed"`

	StimDist string  `desc:"true or false, if the distance between stimuli should be constrianed"`
	MinDist  float64 `desc:"minimum distance between two stimuli"`
	MaxDist  float64 `desc:"maximum distance between two simuli"`

	IgnoreTr bool `desc:"decides if we want to have ignore trials or not"`

	NumStimConst int `desc:"decides how many items the task should be trained on. "`
}

func (ev *SIREnv) Name() string { return ev.Nm }
func (ev *SIREnv) Desc() string { return ev.Dsc }

// SetNStim initializes env for given number of stimuli, init states
func (ev *SIREnv) SetNStim(n int) {
	ev.NStim = n
	ev.Input.SetShape([]int{n}, nil, []string{"N"})
	ev.CtrlInput.SetShape([]int{int(ActionsN)}, nil, []string{"N"})
	ev.Output.SetShape([]int{n}, nil, []string{"N"})
	ev.Reward.SetShape([]int{1}, nil, []string{"1"})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}

	//ev.MinDist = 0 //defined in the main go file
	//ev.MaxDist = 45
}

func (ev *SIREnv) Validate() error {
	if ev.NStim <= 0 {
		return fmt.Errorf("SIREnv: %v NStim == 0 -- must set with SetNStim call", ev.Nm)
	}
	return nil
}

func (ev *SIREnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "CtrlInput":
		return &ev.CtrlInput
	case "Output":
		return &ev.Output
	case "Reward":
		return &ev.Reward
	}
	return nil
}

func (ev *SIREnv) Actions() env.Elements {
	return nil
}

// StimStr returns a letter string rep of stim (A, B...)
func (ev *SIREnv) StimStr(stim int) string {
	return string([]byte{byte('A' + stim)})
}

// String returns the current state as a string
func (ev *SIREnv) String() string {
	//return fmt.Sprintf("%s_%s_mnt1_%s_mnt2_%s_rew_%g", ev.Act, ev.StimStr(ev.Stim), ev.StimStr(ev.Maint1), ev.StimStr(ev.Maint2), ev.Reward.Values[0])
	return fmt.Sprintf("%v_%v_mnt1_%v_mnt2_%v_mnt3_%v_mnt4_%v_rew_%v", ev.Act, ev.Stim, ev.Maint1, ev.Maint2, ev.Maint3, ev.Maint4, ev.Reward.Values[0])
}

func (ev *SIREnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Maint1 = -1
	ev.Maint2 = -1
	ev.Maint3 = -1
}

// SetState sets the input, output states
func (ev *SIREnv) SetState() {
	ev.CtrlInput.SetZeros()
	ev.CtrlInput.Values[ev.Act] = 1
	ev.Input.SetZeros()
	if ev.Act != Recall1 && ev.Act != Recall2 && ev.Act != Recall3 && ev.Act != Recall4 {
		//ev.Input.Values[ev.Stim] = 1
		ev.Input.Values[0] = float64(ev.Stim)
	}
	if ev.Act == Recall1 || ev.Act == Recall2 || ev.Act == Recall3 || ev.Act == Recall4 {
		ev.Input.Values[0] = -999
	}

	ev.Output.SetZeros()
	ev.Output.Values[0] = float64(ev.Stim)
	//ev.Output.Values[ev.Stim] = 1
}

//no point in keeping - netout here is int, but we have float64
// SetReward sets reward based on network's output
//func (ev *SIREnv) SetReward(netout int) bool {
//	cor := ev.Stim // already correct
//	rw := netout == cor
//	if rw {
//		ev.Reward.Values[0] = float64(ev.RewVal)
//	} else {
//		ev.Reward.Values[0] = float64(ev.NoRewVal)
//	}
//	return rw
//}

func (ev *SIREnv) SetRewardThres(netout float64, threshold float64) bool {
	//cor := ev.Stim // already correct
	//threshold := float64(1)
	//here netout is the differnece between the decoded input and decoded output.
	rw := netout <= threshold
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

func (ev *SIREnv) SetRewardCont(netout float64, stim_diff_half float64) bool {
	//here netout is the differnece between the decoded input and decoded output.
	rw := true
	ev.Reward.Values[0] = (stim_diff_half - netout) / (stim_diff_half)
	return rw
}
func (ev *SIREnv) SetRewardContExp(netout float64, denom float64) bool {
	//here netout is the differnece between the decoded input and decoded output.
	rw := true
	ev.Reward.Values[0] = math.Exp(-netout / denom)
	return rw
}

// Step the SIR task
func (ev *SIREnv) StepSIR() {
	for {
		ev.Act = Actions(rand.Intn(int(ActionsN)))
		if ev.Act == Store1 && ev.Maint1 >= 0 { // already full
			continue
		}
		if ev.Act == Recall1 && ev.Maint1 < 0 { // nothing
			continue
		}
		if ev.Act == Store2 && ev.Maint2 >= 0 { // already full
			continue
		}
		if ev.Act == Recall2 && ev.Maint2 < 0 { // nothing
			continue
		}
		if ev.Act == Store3 && ev.Maint3 >= 0 { // already full
			continue
		}
		if ev.Act == Recall3 && ev.Maint3 < 0 { // nothing
			continue
		}
		if ev.Act == Store4 && ev.Maint4 >= 0 { // already full
			continue
		}
		if ev.Act == Recall4 && ev.Maint4 < 0 { // nothing
			continue
		}
		if ev.Act == Ignore && ev.IgnoreTr == false { // we do not want any ignore trials
			continue

		}
		//fmt.Printf("%v", ev.NumStimConst)
		if ev.NumStimConst != 4 {
			if ev.NumStimConst == 3 {
				if ev.Act == Store4 {
					continue
				}
			} else if ev.NumStimConst == 2 {
				if ev.Act == Store4 {
					continue
				}
				if ev.Act == Store3 {
					continue
				}
			} else if ev.NumStimConst == 1 {
				if ev.Act == Store2 {
					continue
				}
				if ev.Act == Store3 {
					continue
				}
				if ev.Act == Store4 {
					continue
				}
			} else {
				log.Fatal("not correct numstimconst")

			}
		}
		//fmt.Printf("%v", ev.Act)
		break
	}
	//ev.Stim = rand.Intn(ev.NStim)
	//ev.Stim = (float64(rand.Intn(ev.NStim))+rand.Float64())
	//ev.Stim = rand.Float64()*float64(3)
	if ev.StimType == "Cont" {
		//ev.Stim = 0.3+rand.Float64()*float64(ev.NStim-1+0.3)
		//ev.Stim = 0.4 + rand.Float64()*float64(3) no ring
		ev.Stim = rand.Float64() * float64(360) //ring

	}
	if ev.StimType == "Fixed" {
		ev.Stim = float64(rand.Intn(ev.NStim))

	}

	if ev.StimDist == "true" {

		fmt.Print("Stim Dist Not implemented for sir4")
		os.Exit(1)

		dist_met := "false"

		for dist_met == "false" {
			if ev.Act == Ignore {
				dist_met = "true"
			}
			if ev.Act == Recall1 {
				dist_met = "true"
			}
			if ev.Act == Recall2 {
				dist_met = "true"
			}
			if ev.Act == Recall3 {
				dist_met = "true"
			}

			if ev.Act == Store1 && ev.Maint2 < 0 && ev.Maint3 < 0 { //current trial is store1 and there is nothing in stimulus 2,3
				dist_met = "true"
			}

			if ev.Act == Store2 && ev.Maint1 < 0 && ev.Maint3 < 0 { //current trial is store2 and there is nothing in stimulus 1,3
				dist_met = "true"
			}

			if ev.Act == Store3 && ev.Maint1 < 0 && ev.Maint2 < 0 { //current trial is store2 and there is nothing in stimulus 1,3
				dist_met = "true"
			}
			if ev.Act == Store1 && ev.Maint2 >= 0 && ev.Maint3 < 0 { //if current trial is store1 and there is something already for stimulus 2, 3 empty

				if math.Abs(ev.Maint2-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint2-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}

			if ev.Act == Store1 && ev.Maint2 < 0 && ev.Maint3 >= 0 { //if current trial is store1 and there is something already for stimulus 3, 2 empty

				if math.Abs(ev.Maint3-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint3-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}

			if ev.Act == Store1 && ev.Maint2 >= 0 && ev.Maint3 >= 0 { //if current trial is store1 and there is something already for stimulus 2 and 3

				if math.Abs(ev.Maint2-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint2-ev.Stim) > ev.MinDist && math.Abs(ev.Maint3-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint3-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}

			if ev.Act == Store2 && ev.Maint1 >= 0 && ev.Maint3 < 0 { //if current trial is store2 and there is something already for stimulus 1, empty 3
				if math.Abs(ev.Maint1-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint1-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}

			if ev.Act == Store2 && ev.Maint1 < 0 && ev.Maint3 >= 0 { //if current trial is store2 and there is something already for stimulus 3, empty 1
				if math.Abs(ev.Maint3-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint3-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}
			if ev.Act == Store2 && ev.Maint1 >= 0 && ev.Maint3 >= 0 { //if current trial is store2 and there is something already for stimulus 1 and 3
				if math.Abs(ev.Maint1-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint1-ev.Stim) > ev.MinDist && math.Abs(ev.Maint3-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint3-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}
			if ev.Act == Store3 && ev.Maint1 >= 0 && ev.Maint2 < 0 { //if current trial is store3 and there is something already for stimulus 1, empty 2
				if math.Abs(ev.Maint1-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint1-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}

			if ev.Act == Store3 && ev.Maint1 < 0 && ev.Maint2 >= 0 { //if current trial is store3 and there is something already for stimulus 2, empty 1
				if math.Abs(ev.Maint2-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint2-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}
			if ev.Act == Store3 && ev.Maint1 >= 0 && ev.Maint2 >= 0 { //if current trial is store3 and there is something already for stimulus 1 and 2
				if math.Abs(ev.Maint1-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint1-ev.Stim) > ev.MinDist && math.Abs(ev.Maint2-ev.Stim) < ev.MaxDist && math.Abs(ev.Maint2-ev.Stim) > ev.MinDist {
					dist_met = "true"
				} else {
					if ev.StimType == "Cont" {
						//ev.Stim = 0.4+rand.Float64()*float64(3) //no ring
						ev.Stim = rand.Float64() * float64(360) //ring
					}
					if ev.StimType == "Fixed" {
						ev.Stim = float64(rand.Intn(ev.NStim))

					}

				}
			}
		}
	}

	switch ev.Act {
	case Store1:
		ev.Maint1 = ev.Stim
	case Store2:
		ev.Maint2 = ev.Stim
	case Store3:
		ev.Maint3 = ev.Stim
	case Store4:
		ev.Maint4 = ev.Stim
	case Ignore:
	case Recall1:
		ev.Stim = ev.Maint1
		ev.Maint1 = -1
	case Recall2:
		ev.Stim = ev.Maint2
		ev.Maint2 = -1
	case Recall3:
		ev.Stim = ev.Maint3
		ev.Maint3 = -1
	case Recall4:
		ev.Stim = ev.Maint4
		ev.Maint4 = -1
	}
	ev.SetState()
}

func (ev *SIREnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	ev.StepSIR()

	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	return true
}

func (ev *SIREnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *SIREnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SIREnv)(nil)
