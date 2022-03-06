// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
sir illustrates the dynamic gating of information into PFC active maintenance, by the basal ganglia (BG). It uses a simple Store-Ignore-Recall (SIR) task, where the BG system learns via phasic dopamine signals and trial-and-error exploration, discovering what needs to be stored, ignored, and recalled as a function of reinforcement of correct behavior, and learned reinforcement of useful working memory representations.
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/popcode" //aneri edit
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/simat"
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/pbwm"
	"github.com/emer/leabra/rl"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no extra learning factors",
				Params: params.Params{
					"Prjn.Learn.Lrate":       "0.02", // slower overall is key
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.WtBal.On":    "false",
				}},
			{Sel: "Layer", Desc: "no decay",
				Params: params.Params{
					"Layer.Act.Init.Decay": "0", // key for all layers not otherwise done automatically
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
			{Sel: ".BgFixed", Desc: "BG Matrix -> GP wiring",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
			{Sel: "RWPrjn", Desc: "Reward prediction -- into PVi",
				Params: params.Params{
					"Prjn.Learn.Lrate": "0.02",
					"Prjn.WtInit.Mean": "0",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
			{Sel: "#Rew", Desc: "Reward layer -- no clamp limits",
				Params: params.Params{
					"Layer.Act.Clamp.Range.Min": "-1",
					"Layer.Act.Clamp.Range.Max": "1",
				}},
			{Sel: ".PFCFmDeep", Desc: "PFC Deep -> PFC fixed",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
			{Sel: ".PFCMntDToOut", Desc: "PFC MntD -> PFC Out fixed",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
			{Sel: ".FmPFCOutD", Desc: "PFC OutD needs to be strong b/c avg act says weak",
				Params: params.Params{
					"Prjn.WtScale.Abs": "4",
				}},
			{Sel: ".PFCFixed", Desc: "Input -> PFC",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8", //turned off to check chunking
					//"Prjn.WtInit.Mean": "0",	//only for chunking checking
					"Prjn.WtInit.Var": "0",
					"Prjn.WtInit.Sym": "false",
				}},
			{Sel: ".MatrixPrjn", Desc: "Matrix learning",
				Params: params.Params{
					"Prjn.Learn.Lrate":         "0.04", // .04 > .1 > .02
					"Prjn.WtInit.Var":          "0.1",
					"Prjn.Trace.GateNoGoPosLR": "1",    // 0.1 default
					"Prjn.Trace.NotGatedLR":    "0.7",  // 0.7 default
					"Prjn.Trace.Decay":         "1.0",  // 1.0 default
					"Prjn.Trace.AChDecay":      "0.0",  // not useful even at .1, surprising..
					"Prjn.Trace.Deriv":         "true", // true default, better than false
				}},
			{Sel: "MatrixLayer", Desc: "exploring these options",
				Params: params.Params{
					"Layer.Act.XX1.Gain":       "100",
					"Layer.Inhib.Layer.Gi":     "2.2", // 2.2 > 1.8 > 2.4
					"Layer.Inhib.Layer.FB":     "1",   // 1 > .5
					"Layer.Inhib.Pool.On":      "true",
					"Layer.Inhib.Pool.Gi":      "2.1", // def 1.9
					"Layer.Inhib.Pool.FB":      "0",
					"Layer.Inhib.Self.On":      "true",
					"Layer.Inhib.Self.Gi":      "0.4", // def 0.3
					"Layer.Inhib.ActAvg.Init":  "0.05",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#GPiThal", Desc: "defaults also set automatically by layer but included here just to be sure",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "1.8", // 1.8 > 2.0
					"Layer.Inhib.Layer.FB":     "1",   // 1.0 > 0.5
					"Layer.Inhib.Pool.On":      "false",
					"Layer.Inhib.ActAvg.Init":  ".2",
					"Layer.Inhib.ActAvg.Fixed": "true",
					"Layer.Act.Dt.GTau":        "3",
					"Layer.Gate.GeGain":        "3",
					"Layer.Gate.NoGo":          "1.25", // was 1 default
					"Layer.Gate.Thr":           "0.25", // .2 default
				}},
			{Sel: "#GPeNoGo", Desc: "GPe is a regular layer -- needs special params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "2.4", // 2.4 > 2.2 > 1.8 > 2.6
					"Layer.Inhib.Layer.FB":     "0.5",
					"Layer.Inhib.Layer.FBTau":  "3", // otherwise a bit jumpy
					"Layer.Inhib.Pool.On":      "false",
					"Layer.Inhib.ActAvg.Init":  ".2",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: ".PFC", Desc: "pfc defaults",
				Params: params.Params{
					"Layer.Act.XX1.Gain":       "5", //gain should be lower to make it less blocky?what is the trade off?
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.Pool.On":      "true",
					"Layer.Inhib.Pool.Gi":      "1.8",
					"Layer.Inhib.Pool.FB":      "1",
					"Layer.Inhib.ActAvg.Init":  "0.2",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#Input", Desc: "Basic params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":  "0.25",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#Output", Desc: "Basic params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "2",
					"Layer.Inhib.Layer.FB":     "0.5",
					"Layer.Inhib.ActAvg.Init":  "0.25",
					"Layer.Inhib.ActAvg.Fixed": "true",
				}},
			{Sel: "#InputToOutput", Desc: "weaker",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.5", //turned off just for chunking.

					//"Prjn.Learn.Learn": "false", //only for checking chunking
					//"Prjn.WtInit.Mean": "0",//only for checking chunking
					//"Prjn.WtInit.Var":  "0",//only for checking chunking

				}},
			{Sel: "#Hidden", Desc: "Basic params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2",
					"Layer.Inhib.Layer.FB": "0.5",
				}},

			{Sel: "#SNc", Desc: "allow negative",
				Params: params.Params{
					"Layer.Act.Clamp.Range.Min": "-1",
					"Layer.Act.Clamp.Range.Max": "1",
				}},
			{Sel: "#RWPred", Desc: "keep it guessing",
				Params: params.Params{
					"Layer.PredRange.Min": "0.05", // single most important param!  was .01 -- need penalty..
					"Layer.PredRange.Max": "0.95",
				}},
			{Sel: "#Chunk", Desc: "Basic params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
					"Layer.Act.XX1.Gain":   "5", //gain should be lower to make it less blocky?what is the trade off?
				}},
			{Sel: "#InputToChunk", Desc: "Input to Chunk",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.7",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
			{Sel: ".PFCFixedChunk", Desc: "Chunk -> PFC",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
				}},
			{Sel: ".PFCMntDChunk", Desc: "PFC MntD -> Chunk fixed",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.2",
					"Prjn.WtInit.Var":  "0",
					"Prjn.WtInit.Sym":  "false",
					"Prjn.WtScale.Rel": "3.5",
				}},
			//{Sel: "#ChunkToOutput", Desc: "Chunk to Output, weaker", //only for checking chunking
			//	Params: params.Params{//only for checking chunking
			//		"Prjn.WtScale.Rel": "0.5",//only for checking chunking
			//	}},//only for checking chunking

		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	BurstDaGain float32           `desc:"strength of dopamine bursts: 1 default -- reduce for PD OFF, increase for PD ON"`
	DipDaGain   float32           `desc:"strength of dopamine dips: 1 default -- reduce to siulate D2 agonists"`
	Net         *pbwm.Network     `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	TrnEpcLog   *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog   *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog   *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	RunLog      *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats    *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	SimMat      *simat.SimMat     `view:"no-inline" desc:"similarity matrix"`
	Params      params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet    string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag         string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	MaxRuns     int               `desc:"maximum number of model runs to perform"`
	MaxEpcs     int               `desc:"maximum number of epochs to run per model run"`
	MaxTrls     int               `desc:"maximum number of training trials per epoch"`
	NZeroStop   int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv    SIREnv            `desc:"Training environment -- SIR environment"`
	TestEnv     SIREnv            `desc:"Testing nvironment -- SIR environment"`
	Time        leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn      bool              `desc:"whether to update the network view while running"`
	TrainUpdt   leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt    leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TstRecLays  []string          `desc:"names of layers to record activations etc of during testing"`

	InOneTsr *etensor.Float32 `view:"-" desc:"for holding layer values"`
	InTwoTsr *etensor.Float32 `view:"-" desc:"for holding layer values"`

	TmpVals    []float32 `view:"-" desc:"for holding decoded layer values"`
	TmpVals2   []float32 `view:"-" desc:"for holding decoded layer values"`
	TmpValsInp []float32 `view:"-" desc:"for holding decoded layer values"`
	TmpValsCh  []float32 `view:"-" desc:"for holding decoded layer values"`
	TmpValsPFC []float32 `view:"-" desc:"for holding decoded layer values"`

	pop_min   float32 `desc:"minimum value representable -- for GaussBump, typically include extra to allow mean with activity on either side to represent the lowest value you want to encode"`
	pop_max   float32 `desc:"minimum value representable -- for GaussBump, typically include extra to allow mean with activity on either side to represent the lowest value you want to encode"`
	pop_sigma float64 `def:"0.2" viewif:"Code=GaussBump" desc:"sigma parameter of a gaussian specifying the tuning width of the coarse-coded units, in normalized 0-1 range, float64 so can use tags, otherwise inside setrange, need to be float32 so will be changed before inputting there"`

	RewThreshold float64 `desc: "threshold for reward function"`

	TrlDA         float64 `inactive:"+" desc:"dopamine level on this trial"`
	TrlAbsDA      float64 `inactive:"+" desc:"absolute value of dopamine on this trial"`
	TrlRewPred    float64 `inactive:"+" desc:"reward prediction level on this trial"`
	TrlErr        float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"`
	TrlSSE        float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE     float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff    float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcDA         float64 `inactive:"+" desc:"last epoch's average dopamine"`
	EpcAbsDA      float64 `inactive:"+" desc:"last epoch's avg abs dopamine"`
	EpcRewPred    float64 `inactive:"+" desc:"last epoch's avg rew pred dopamine"`
	EpcSSE        float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero SSE"`

	Lesion        string  `inactive:"+" desc:"what type of lesion: ex) PFC, Hidden, None, PFCHidden"`
	LesionProp    float64 `inactive:"+" desc:"proportion of test trials to have a lesion"`
	LesionApplied string  `inactive:"+" desc:"if lesion is actually applied"`
	Folder        string  `inactive:"+" desc:"folder for saving"`
	LayerSize     int     `inactive:"+" desc:"layer size"`
	Stripes       int     `inactive:"+" desc:"number of pfc stripes"`
	Experiment    string  `inactive:"+" desc:"type of experiment to run"`
	NumModels     int     `inactive:"+" desc:"number of models to run"`

	// internal state - view:"-"
	SumDA        float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAbsDA     float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumRewPred   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumErr       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	WtsGrid      *etview.TensorGrid          `view:"-" desc:"the weights grid view"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	TstTrlFile   *os.File                    `view: "-" desc: "log file"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64                       `view:"-" desc:"the current random seed"`
	LastEpcTime  time.Time                   `view:"-" desc:"timer for last epoch"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &pbwm.Network{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.SimMat = &simat.SimMat{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle //leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.TstRecLays = []string{"Input", "Output", "Chunk", "GPiThal", "PFCmntD", "PFCoutD"}
	ss.Defaults()
}

func (ss *Sim) Defaults() {
	ss.BurstDaGain = 1
	ss.DipDaGain = 1
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigRunLog(ss.RunLog)
	ss.ConfigInTsrs()

}

func (ss *Sim) ConfigInTsrs() { //need this to not have error nill pointer error

	if ss.InOneTsr == nil {

		ss.InOneTsr = etensor.NewFloat32([]int{1, ss.LayerSize}, nil, nil)

	}

	if ss.InTwoTsr == nil {

		ss.InTwoTsr = etensor.NewFloat32([]int{1, ss.LayerSize}, nil, nil)

	}

}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 100
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 500
		ss.NZeroStop = 5
	}
	if ss.MaxTrls == 0 { // allow user override
		ss.MaxTrls = 100
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.SetNStim(4)
	ss.TrainEnv.RewVal = 1
	ss.TrainEnv.NoRewVal = 0
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TrainEnv.Trial.Max = ss.MaxTrls
	ss.TrainEnv.StimType = "Cont" //continuous (0-3) vs. fixed stimulus (0,1,2,3)
	ss.TrainEnv.StimDist = "false" //will be defined in the tag
	ss.TrainEnv.MaxDist = 45
	ss.TrainEnv.MinDist = 0

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.SetNStim(4)
	ss.TestEnv.RewVal = 1
	ss.TestEnv.NoRewVal = 0
	ss.TestEnv.Validate()
	ss.TestEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually
	ss.TestEnv.Trial.Max = 700      // good amount for testing
	ss.TestEnv.StimType = "Cont"
	ss.TestEnv.StimDist = "false" //will be defined in the tag
	ss.TestEnv.MaxDist = 45
	ss.TestEnv.MinDist = 0

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)

	ss.pop_min = 0      //ring
	ss.pop_max = 360    //ring
	ss.pop_sigma = 0.15 //ring

	//ss.pop_min = 0 //no ring
	//ss.pop_max = 3.8 //no ring
	//ss.pop_sigma = 0.15 // need to optimize over this parameter //no ring

	ss.RewThreshold = 10

	ss.Lesion = "None"
	ss.LesionProp = 0
	ss.LesionApplied = "no"

	ss.LayerSize = 20
	ss.Stripes = 2
}

func (ss *Sim) ConfigNet(net *pbwm.Network) {
	net.InitName(net, "SIR")
	rew, rp, da := rl.AddRWLayers(&net.Network, "", relpos.Behind, 2)
	snc := da.(*rl.RWDaLayer)
	snc.SetName("SNc")

	inp := net.AddLayer2D("Input", 1, ss.LayerSize, emer.Input)
	ctrl := net.AddLayer2D("CtrlInput", 1, 5, emer.Input)
	out := net.AddLayer2D("Output", 1, ss.LayerSize, emer.Target)
	hid := net.AddLayer2D("Hidden", ss.LayerSize, ss.LayerSize, emer.Hidden)
	chunk := net.AddLayer2D("Chunk", 1, ss.LayerSize, emer.Hidden)
	inp.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: rew.Name(), YAlign: relpos.Front, XAlign: relpos.Left})
	//out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 1})
	out.SetRelPos(relpos.Rel{Rel: relpos.LeftOf, Other: "Input", YAlign: relpos.Front, Space: 1})
	ctrl.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "Input", XAlign: relpos.Left, Space: 2})
	hid.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "CtrlInput", XAlign: relpos.Left, Space: 2})
	chunk.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "Hidden", XAlign: relpos.Left, Space: 3})

	// args: nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX
	//mtxGo, mtxNoGo, gpe, gpi, cini, pfcMnt, pfcMntD, pfcOut, pfcOutD := net.AddPBWM("", 4, 2, 2, 1, 5, 1, ss.LayerSize)
	//mtxGo, mtxNoGo, gpe, gpi, cini, pfcMnt, pfcMntD, pfcOut, pfcOutD := net.AddPBWM("", 8, 1, 1, 1, 5, 1, ss.LayerSize)
	mtxGo, mtxNoGo, gpe, gpi, cini, pfcMnt, pfcMntD, pfcOut, pfcOutD := net.AddPBWM("", ss.Stripes, 1, 1, 1, 5, 1, ss.LayerSize)
	_ = gpe
	_ = gpi
	_ = pfcMnt
	_ = pfcMntD
	_ = pfcOut

	fmt.Printf("stripes %v", ss.Stripes)

	cin := cini.(*pbwm.CINLayer)
	cin.RewLays.Add(rew.Name(), rp.Name())

	//mtxGo.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Rew", YAlign: relpos.Front, Space: 14})
	mtxGo.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Rew", YAlign: relpos.Front, Space: 20})

	full := prjn.NewFull()
	fmin := prjn.NewRect()
	fmin.Size.Set(1, 1)
	fmin.Scale.Set(1, 1)
	fmin.Wrap = true

	//correct projection
	fmin2 := prjn.NewRect()
	//fmin2.Size.Set(1, 8)
	fmin2.Size.Set(1, ss.Stripes)
	fmin2.Scale.Set(1, 1)
	fmin2.Wrap = true

	//trial projection, projects just to bottom stripe.
	fminbot := prjn.NewRect()
	fminbot.Size.Set(1, 1)
	fminbot.Scale.Set(1, 1)
	fminbot.Wrap = false

	//trial projection, projects just to top stripe (s).
	fmintop := prjn.NewRect()
	fmintop.Size.Set(1, 1)
	fmintop.Scale.Set(1, 1)
	fmintop.RecvStart.Set(0, 1)
	fmintop.Wrap = true

	net.ConnectLayersPrjn(ctrl, rp, full, emer.Forward, &rl.RWPrjn{})
	net.ConnectLayersPrjn(pfcMntD, rp, full, emer.Forward, &rl.RWPrjn{})
	net.ConnectLayersPrjn(pfcOutD, rp, full, emer.Forward, &rl.RWPrjn{})

	pj := net.ConnectLayersPrjn(ctrl, mtxGo, fmin, emer.Forward, &pbwm.MatrixTracePrjn{})
	pj.SetClass("MatrixPrjn")
	pj = net.ConnectLayersPrjn(ctrl, mtxNoGo, fmin, emer.Forward, &pbwm.MatrixTracePrjn{})
	pj.SetClass("MatrixPrjn")
	pj = net.ConnectLayers(inp, pfcMnt, fmintop, emer.Forward) //hybrid model
	//pj = net.ConnectLayers(inp, pfcMnt, fmin, emer.Forward) //original model
	pj.SetClass("PFCFixed")

	net.ConnectLayers(inp, hid, full, emer.Forward) //turned off for checking chunking
	//net.ConnectLayers(chunk, out, full, emer.Forward) //just for checking chunking

	net.ConnectLayers(ctrl, hid, full, emer.Forward)
	net.BidirConnectLayers(hid, out, full)
	//pj = net.ConnectLayers(pfcOutD, hid, full, emer.Forward)
	pj = net.ConnectLayers(pfcOutD, hid, fmin2, emer.Forward)
	pj.SetClass("FmPFCOutD")
	//pj = net.ConnectLayers(pfcOutD, out, full, emer.Forward)
	pj = net.ConnectLayers(pfcOutD, out, fmin2, emer.Forward)
	pj.SetClass("FmPFCOutD")
	net.ConnectLayers(inp, out, full, emer.Forward)

	pj = net.ConnectLayers(inp, chunk, fmin, emer.Forward)
	//pj=net.ConnectLayers(pfcMntD,chunk,fmin2,emer.Forward) //original model
	pj = net.ConnectLayers(pfcMntD, chunk, fminbot, emer.Forward) //hybrid model
	pj.SetClass("PFCMntDChunk")
	pj = net.ConnectLayers(chunk, pfcMnt, fminbot, emer.Forward) //hybrid model
	//pj = net.ConnectLayers(chunk, pfcMnt, fmin, emer.Forward) //original model
	pj.SetClass("PFCFixedChunk")

	snc.SendDA.AddAllBut(net, nil) // send dopamine to all layers..

	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.NewRun()
	ss.UpdateView(true)
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.String())
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.String())
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
		if qtr == 2 {
			ss.ApplyReward(train)
		}
	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	//pc:= popcode.OneD{} //no ring
	//pc.Defaults() //no ring

	//pc.SetRange(ss.pop_min,ss.pop_max,float32(ss.pop_sigma)) //no ring

	pc := popcode.Ring{}             //ring
	pc.Defaults()                    //ring
	pc.Min = ss.pop_min              //ring
	pc.Max = ss.pop_max              //ring
	pc.Sigma = float32(ss.pop_sigma) //ring

	lays := []string{"Input", "CtrlInput", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats == nil {
			continue
		}

		if lnm == "Input" {
			v := float32(pats.FloatVal1D(0))

			if v != -999 {

				//pc := OneD{}
				//pc.Defaults()
				//var vals []float32
				//pc.Values(&vals, 11)
				// fmt.Printf("vals: %v\n", vals)

				//corVals := []float32{-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5}

				//CmprFloats(vals, corVals, "vals for 11 units", t)
				//var pat []float32
				//pc.Encode(&pat, 0.5, 11, Set)

				//var temp []float32

				//fmt.Printf("temp is %v", temp)
				//pc.Encode(&temp,v,ss.LayerSize,Set==true)
				//&ss.InOneTsr.Values = temp
				//pc.Encode(&ss.InOneTsr.Values,v,ss.LayerSize,Set)

				//pc.Encode(&ss.InOneTsr.Values,v,ss.LayerSize,false) //original command //no ring
				pc.Encode(&ss.InOneTsr.Values, v, ss.LayerSize) //ring

				ly.ApplyExt(ss.InOneTsr)
			}

			//fmt.Printf("pats is %v",pats)
			//fmt.Printf("InOneTsr is %v",ss.InOneTsr)
		}
		if lnm == "CtrlInput" {
			ly.ApplyExt(pats) //no popcode

		}
		if lnm == "Output" {
			v := float32(pats.FloatVal1D(0))
			//pc.Encode(&ss.InTwoTsr.Values,v,ss.LayerSize,false) //no ring
			pc.Encode(&ss.InTwoTsr.Values, v, ss.LayerSize) //ring
			ly.ApplyExt(ss.InTwoTsr)

		}

		//ly.ApplyExt(pats)
	}
}

func sumarray(array []float32) float32 {
	resultsum := float32(0)
	for _, v := range array {
		resultsum += float32(math.Abs(float64(v)))
	}

	return resultsum
}
func diff(a, b []float32) []float32 {
	result := []float32{}
	for x, _ := range b {
		result = append(result, a[x]-b[x])
	}
	return result
}

// ApplyReward computes reward based on network output and applies it -- call
// at start of 3rd quarter (plus phase)
func (ss *Sim) ApplyReward(train bool) {
	var en *SIREnv
	if train {
		en = &ss.TrainEnv
	} else {
		en = &ss.TestEnv
	}
	if en.Act != Recall1 && en.Act != Recall2 { // only reward on recall trials!
		return
	}
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	//pc := popcode.OneD{}//previously defined pc does not work here //no ring
	//pc.Defaults() //no ring
	//pc.SetRange(ss.pop_min,ss.pop_max,float32(ss.pop_sigma)) //does not say to add these two - lets see if it works //no ring

	pc := popcode.Ring{}             //ring
	pc.Defaults()                    //ring
	pc.Min = ss.pop_min              //ring
	pc.Max = ss.pop_max              //ring
	pc.Sigma = float32(ss.pop_sigma) //ring

	//Actual Value
	out.UnitVals(&ss.TmpVals, "ActM") //writes ActM value from the layer
	//TARGET
	out.UnitVals(&ss.TmpVals2, "Targ") //writes Targ value from the layer

	//mxi := out.Pools[0].Inhib.Act.MaxIdx
	//en.SetReward(mxi)

	en.SetRewardThres(float64(sumarray(diff(ss.TmpVals2, ss.TmpVals))), ss.RewThreshold) //based on difference + threshold (in sir_env)

	pats := en.State("Reward")
	ly := ss.Net.LayerByName("Rew").(leabra.LeabraLayer).AsLeabra()
	ly.ApplyExt1DTsr(pats)
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {

	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumDA = 0
	ss.SumAbsDA = 0
	ss.SumRewPred = 0
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.SumErr = 0
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlDA = 0
	ss.TrlAbsDA = 0
	ss.TrlRewPred = 0
	ss.TrlErr = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcDA = 0
	ss.EpcAbsDA = 0
	ss.EpcRewPred = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) {
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	snc := ss.Net.LayerByName("SNc").(leabra.LeabraLayer).AsLeabra()
	rp := ss.Net.LayerByName("RWPred").(leabra.LeabraLayer).AsLeabra()
	ss.TrlDA = float64(snc.Neurons[0].Act)
	ss.TrlAbsDA = math.Abs(ss.TrlDA)
	ss.TrlRewPred = float64(rp.Neurons[0].Act)
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = out.MSE(0.05) // 0.5 = per-unit tolerance -- right side of .5 //originally 0.5; new 0.05
	//fmt.Printf("TrlSSE %v, TrlAvgSSE, %v", ss.TrlSSE, ss.TrlAvgSSE)
	if ss.TrlSSE > 0 {
		ss.TrlErr = 1
	} else {
		ss.TrlErr = 0
	}
	if accum {
		ss.SumDA += ss.TrlDA
		ss.SumAbsDA += ss.TrlAbsDA
		ss.SumRewPred += ss.TrlRewPred
		ss.SumErr += ss.TrlErr
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
	}
	//fmt.Printf("SumSSE %v, SumAvgSSE, %v", ss.SumSSE, ss.SumAvgSSE)
	return
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {

	var err error
	//fnm := ss.LogFileName("epc")
	fnm := "/gpfs/home/asoni4/leabra/examples/workingmemory/sir2_chunk/results/" + ss.Folder + ss.RunName() + "EpcLog.csv"
	ss.TrnEpcFile, err = os.Create(fnm)

	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()

	if err != nil {
		log.Println(err)
		ss.TrnEpcFile = nil
	} else {
		//fmt.Printf("Saving epoch log to: %s\n", fnm)
		defer ss.TrnEpcFile.Close()
	}
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

func (ss *Sim) UnLesionNet(net *pbwm.Network) {
	net.LayersSetOff(false)
	net.UnLesionNeurons()
	net.InitActs()
}

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	if ss.Lesion != "None" {
		comp := rand.Float64()
		if comp < ss.LesionProp {
			ss.LesionApplied = "yes"

			if ss.Lesion == "PFC" {
				ss.Net.LayerByName("PFCoutD").SetOff(true)
			}
			if ss.Lesion == "Hidden" {
				ss.Net.LayerByName("Hidden").SetOff(true)
			}
			if ss.Lesion == "PFCHidden" {
				ss.Net.LayerByName("PFCoutD").SetOff(true)
				ss.Net.LayerByName("Hidden").SetOff(true)
			}
		}

	}

	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
	if ss.LesionApplied == "yes" {
		ss.UnLesionNet(ss.Net)
		ss.LesionApplied = "no"
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TestEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg, don't present
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {

	var err error

	//fnm := "C:/Users/Aneri/go/src/leabra/examples/sir_proj/sir2_new/results/"+ss.RunName()+".csv"

	fnm := "/gpfs/home/asoni4/leabra/examples/workingmemory/sir2_chunk/results/" + ss.Folder + ss.RunName() + ".csv"
	ss.TstTrlFile, err = os.Create(fnm)

	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()

	if err != nil {
		log.Println(err)
		ss.TstTrlFile = nil
	} else {
		fmt.Printf("Saving testtrial log to: %v\n", fnm)
		defer ss.TstTrlFile.Close()
	}
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}

	// gpi := ss.Net.LayerByName("GPiThal").(*pbwm.GPiThalLayer)
	// gpi.Gate.Thr = 0.5 // todo: these are not taking in params
	// gpi.Gate.NoGo = 0.4
	//
	matg := ss.Net.LayerByName("MatrixGo").(*pbwm.MatrixLayer)
	matn := ss.Net.LayerByName("MatrixNoGo").(*pbwm.MatrixLayer)

	matg.Matrix.BurstGain = ss.BurstDaGain
	matg.Matrix.DipGain = ss.DipDaGain
	matn.Matrix.BurstGain = ss.BurstDaGain
	matn.Matrix.DipGain = ss.DipDaGain

	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TrainEnv" etc
	return err
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts.gz"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv         // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Trial.Max) // number of trials in view

	ss.EpcDA = ss.SumDA / nt
	ss.SumDA = 0
	ss.EpcAbsDA = ss.SumAbsDA / nt
	ss.SumAbsDA = 0
	ss.EpcRewPred = ss.SumRewPred / nt
	ss.SumRewPred = 0
	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.SumErr) / nt
	ss.SumErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("DA", row, ss.EpcDA)
	dt.SetCellFloat("AbsDA", row, ss.EpcAbsDA)
	dt.SetCellFloat("RewPred", row, ss.EpcRewPred)
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"DA", etensor.FLOAT64, nil, nil},
		{"AbsDA", etensor.FLOAT64, nil, nil},
		{"RewPred", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DA", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("AbsDA", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("RewPred", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TestEnv.Epoch.Prv // this is triggered by increment so use previous value

	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	chunk := ss.Net.LayerByName("Chunk").(leabra.LeabraLayer).AsLeabra()
	pfc := ss.Net.LayerByName("PFCmntD").(leabra.LeabraLayer).AsLeabra()

	//pc := popcode.OneD{}//previously defined pc does not work here //no ring
	//pc.Defaults() //no ring
	//pc.SetRange(ss.pop_min,ss.pop_max,float32(ss.pop_sigma)) //does not say to add these two //no ring

	pc := popcode.Ring{}             //ring
	pc.Defaults()                    //ring
	pc.Min = ss.pop_min              //ring
	pc.Max = ss.pop_max              //ring
	pc.Sigma = float32(ss.pop_sigma) //ring

	trl := ss.TestEnv.Trial.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TestEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.String())
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	dt.SetCellFloat("DA", row, ss.TrlDA)
	dt.SetCellFloat("AbsDA", row, ss.TrlAbsDA)
	dt.SetCellFloat("RewPred", row, ss.TrlRewPred)

	for _, lnm := range ss.TstRecLays {
		tsr := ss.ValsTsr(lnm)
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ly.UnitValsTensor(tsr, "ActM")
		dt.SetCellTensor(lnm, row, tsr)
	}

	out.UnitVals(&ss.TmpVals, "ActM")
	outdecode := pc.Decode(ss.TmpVals)
	dt.SetCellFloat("OutDecode", row, float64(outdecode))

	out.UnitVals(&ss.TmpVals2, "Targ")   //writes Act value from the layer
	outdecode2 := pc.Decode(ss.TmpVals2) //this decodes the slide into a single float32
	dt.SetCellFloat("OutTarget", row, float64(outdecode2))

	inp.UnitVals(&ss.TmpValsInp, "Act")
	indecode := pc.Decode(ss.TmpValsInp)
	dt.SetCellFloat("InDecode", row, float64(indecode))

	chunk.UnitVals(&ss.TmpValsCh, "Act")
	chdecode := pc.Decode(ss.TmpValsCh)
	dt.SetCellFloat("ChunkDecode", row, float64(chdecode))

	pfc.UnitVals(&ss.TmpValsPFC, "Act")
	for stripe := 0; stripe < ss.Stripes; stripe++ {
		pfcdecode := pc.Decode(ss.TmpValsPFC[stripe*ss.LayerSize : (stripe+1)*ss.LayerSize])
		stnm := "stripe" + string(stripe)
		dt.SetCellFloat(stnm, row, float64(pfcdecode))
	}

	dt.SetCellString("Lesion", row, ss.Lesion)
	dt.SetCellFloat("LesionProp", row, ss.LesionProp)
	dt.SetCellString("LesionApplied", row, ss.LesionApplied)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()

	if ss.TstTrlFile != nil {
		if ss.TestEnv.Trial.Cur == 0 {
			dt.WriteCSVHeaders(ss.TstTrlFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TstTrlFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Trial.Max
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"DA", etensor.FLOAT64, nil, nil},
		{"AbsDA", etensor.FLOAT64, nil, nil},
		{"RewPred", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{lnm, etensor.FLOAT64, ly.Shp.Shp, nil})
	}

	for stripe := 0; stripe < ss.Stripes; stripe++ {
		stnm := "stripe" + string(stripe)
		sch = append(sch, etable.Column{stnm, etensor.FLOAT64, nil, nil}) //adds pfcdecode value to table
	}

	sch = append(sch, etable.Schema{

		{"OutDecode", etensor.FLOAT64, nil, nil},    //adds outdecode value to table
		{"OutTarget", etensor.FLOAT64, nil, nil},    //adds target value to table
		{"InDecode", etensor.FLOAT64, nil, nil},     //adds input value to table
		{"ChunkDecode", etensor.FLOAT64, nil, nil},  //adds chunk value to table
		{"Lesion", etensor.STRING, nil, nil},        //adds lesion type
		{"LesionProp", etensor.FLOAT64, nil, nil},   //adds prop of trials with lesion
		{"LesionApplied", etensor.STRING, nil, nil}, //add if lesion was actually applied in that test trial
	}...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DA", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("AbsDA", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("RewPred", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TstRecLays {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// trl := ss.TstTrlLog
	// tix := etable.NewIdxView(trl)
	epc := ss.TestEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TestEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName()

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()

	//	labs := []string{"    A B C D ", "  A B C D   A B C D ", "  A B C D  A B C D",
	//		"A B C D  A B C D ", "A B C D  A B C D", "   A B C D ", "  S1 S2 I R1 R2 "}
	//	nv.ConfigLabels(labs)
	//
	//	lays := []string{"Input", "PFCmnt", "PFCmntD", "PFCout", "PFCoutD", "Output", "CtrlInput"}
	//
	//	for li, lnm := range lays {
	//		ly := nv.LayerByName(lnm)
	//		lbl := nv.LabelByName(labs[li])
	//		lbl.Pose = ly.Pose
	//		lbl.Pose.Pos.Y += .08
	//		lbl.Pose.Pos.Z += .02
	//		lbl.Pose.Scale.SetMul(mat32.Vec3{1, 0.3, 0.5})
	//	}
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("sir")
	gi.SetAppAbout(`illustrates the dynamic gating of information into PFC active maintenance, by the basal ganglia (BG). It uses a simple Store-Ignore-Recall (SIR) task, where the BG system learns via phasic dopamine signals and trial-and-error exploration, discovering what needs to be stored, ignored, and recalled as a function of reinforcement of correct behavior, and learned reinforcement of useful working memory representations. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch10/sir/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("sir", "SIR: PBWM", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't break on chg
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "update", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch10/sir/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxRuns, "runs", 100, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", false, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", false, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.StringVar(&ss.Lesion, "Lesion", "None", "lesion type")
	flag.Float64Var(&ss.LesionProp, "LesionProp", 0, "proportion of test trials with lesion")
	flag.Float64Var(&ss.pop_sigma, "pop_sigma", 0.15, "sigma for pop coding")
	flag.Float64Var(&ss.RewThreshold, "RewThreshold", 10, "threshold for rew in sir2_env")
	flag.StringVar(&ss.Folder, "folder", "", "folder for saving results")
	flag.StringVar(&ss.Experiment, "experiment", "", "experiment name")
	flag.IntVar(&ss.NumModels, "NumModels", 5, "number of models to run")
	flag.StringVar(&ss.TrainEnv.StimDist, "TrainStimDist", "false", "restrict whether or  not we choose narrow stim, should be true or false, for train")
	flag.StringVar(&ss.TestEnv.StimDist, "TestStimDist", "false", "restrict whether or  not we choose narrow stim, should be true or false, for test")
	//	flag.IntVar(&ss.Stripes, "Stripes",2,"Number of PFC Stripes")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	models := make([]int, ss.NumModels)
	for i := 0; i < ss.NumModels; i++ {
		models[i] = i
	}
	//fmt.Printf("%v", models)

	if ss.Experiment == "Lesion" {

		//Lesion experiments
		props := []float64{0, 0.2, 0.4, 0.6, 0.8, 1} //proportion of lesions
		for _, w := range props {
			ss.LesionProp = w
			fmt.Printf("LesionProps is %v", w)

			for _, v := range models {
				ss.Tag = "model" + strconv.Itoa(v) + "_Lesion" + ss.Lesion + "_LesionProp" + strconv.FormatFloat(ss.LesionProp, 'G', -1, 64)
				fmt.Printf(ss.Tag)
				ss.TrainRun()
				ss.RunTestAll()
			}
		}

	} else if ss.Experiment == "RewThres" {

		RewThresholds := []float64{0.5, 1, 1.5, 2, 2.5, 3, 3.5}

		for _, w := range RewThresholds {

			ss.RewThreshold = w
			fmt.Printf("reward threshold is %v", w)
			for _, v := range models {

				ss.Tag = "model" + strconv.Itoa(v) + "_Threhsold" + strconv.FormatFloat(ss.RewThreshold, 'G', -1, 64) + "_sigma" + strconv.FormatFloat(float64(ss.pop_sigma), 'G', -1, 32)
				fmt.Printf(ss.Tag)

				ss.TrainRun()
				ss.RunTestAll()

			}
		}
	} else {

		for _, v := range models {
			//	//ss.Tag = "model"+strconv.Itoa(v)+"_Lesion"+ss.Lesion+"_LesionProp"+strconv.FormatFloat(ss.LesionProp,'G',-1,64)
			ss.Tag = "hybridmodel" + strconv.Itoa(v) + "_RewThreshold" + strconv.FormatFloat(ss.RewThreshold, 'G', -1, 64)
			//ss.Tag = "model"+strconv.Itoa(v)
			fmt.Printf(ss.Tag)
			ss.TrainRun()
			ss.RunTestAll()
		}
	}

	//used to optimize reward threshold.

	//	models := []int{0, 1, 2, 3, 4}

	//	for _,v := range models {
	//	//	ss.Tag = "model"+strconv.Itoa(v)+"_Threhsold"+strconv.FormatFloat(ss.RewThreshold,'G',-1,64)+"_sigma"+strconv.FormatFloat(float64(ss.pop_sigma),'G',-1,32)
	//		ss.Tag = "model"+strconv.Itoa(v)+ss.Tag
	//		fmt.Printf(ss.Tag)
	//		ss.TrainRun()
	//		ss.RunTestAll()
	//
	//	}

	if saveEpcLog {
		var err error
		//fnm := ss.LogFileName("epc")
		fnm := "/gpfs/home/asoni4/leabra/examples/workingmemory/sir2_chunk/results/" + ss.Folder + ss.RunName() + "EpcLog.csv"
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %s\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %s\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)

}
