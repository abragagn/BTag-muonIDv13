#include <iostream>
#include <sstream>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>

#include "PDAnalyzer.h"

#include "TDirectory.h"
#include "TBranch.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLorentzVector.h"
#include "TFile.h"

// additional features
#include "PDSecondNtupleWriter.h"
#include "PDMuonVar.cc"
#include "PDSoftMuonMvaEstimator.cc"
#include "AlbertoUtil.cc"

using namespace std;

//pdTreeAnalyze /lustre/cmswork/abragagn/ntuList/MC2017Lists/BsToJpsiPhi_2017_DCAP.list hist.root -v outputFile ntu.root -v histoMode RECREATE -v use_gen t -v process BsJPsiPhi -n 10000

PDAnalyzer::PDAnalyzer() {

    std::cout << "new PDAnalyzer" << std::endl;

    // user parameters are set as names associated to a string, 
    // default values can be set in the analyzer class contructor

    setUserParameter( "verbose", "f" );
    setUserParameter( "minPtMuon", "2." );
    setUserParameter( "maxEtaMuon", "2.4" );
    setUserParameter( "outputFile", "ntu.root" );
    setUserParameter( "process", "BsJPsiPhi" );

    setUserParameter( "ptCut", "40.0" ); //needed for paolo's code for unknow reasons

}


PDAnalyzer::~PDAnalyzer() {
}



void PDAnalyzer::beginJob() {

    PDAnalyzerUtil::beginJob();

    // user parameters are retrieved as strings by using their names;
    // numeric parameters ( int, float or whatever ) can be directly set
    // by passing the corresponding variable,
    // e.g. getUserParameter( "name", x )

    getUserParameter( "verbose", verbose );
    getUserParameter( "minPtMuon", minPtMuon );
    getUserParameter( "maxEtaMuon", maxEtaMuon );
    getUserParameter( "process", process );

    getUserParameter( "ptCut", ptCut ); //needed for paolo's code for unknow reasons

    tWriter = new PDSecondNtupleWriter;
    tWriter->open( getUserParameter("outputFile"), "RECREATE" ); // second ntuple

    nReal = 0;
    nFake = 0;
    nB = 0;

    counter = new int[10];
    for(int i=0; i<10; ++i) counter[i]=0;

    return;

}


void PDAnalyzer::book() {

    // putting "autoSavedObject" in front of the histo creation 
    // it's automatically marked for saving on file; the option 
    // is uneffective when not using the full utility

    return;

}


void PDAnalyzer::reset() {
// automatic reset
    autoReset();
    return;
}


bool PDAnalyzer::analyze( int entry, int event_file, int event_tot ) {

    if ( verbose ) {
        cout << " +++++++++++++++++++++++++++ " << endl;
        cout << "entry: "
                 << entry << " " << event_file << " " << event_tot << endl;
        cout << "run: " << runNumber << " , "
                 << "evt: " << eventNumber << endl;
    }
    else {

        if ( (!(event_tot%10) && event_tot<100 ) || 
     (!(event_tot %100) && event_tot<1000 ) || 
     (!(event_tot %1000) && event_tot<10000 ) || 
     (!(event_tot %10000) && event_tot<100000 ) || 
     (!(event_tot %100000) && event_tot<1000000 ) || 
     (!(event_tot %1000000) && event_tot<10000000 ) )
            cout << " == at event " << event_file << " " << event_tot << endl;
    }

    // additional features
    computeMuonVar();
    tWriter->Reset();
    convSpheCart(jetPt, jetEta, jetPhi, jetPx, jetPy, jetPz);
    convSpheCart(muoPt, muoEta, muoPhi, muoPx, muoPy, muoPz);
    convSpheCart(trkPt, trkEta, trkPhi, trkPx, trkPy, trkPz);

    //SEARCHING SIGNAL
    int signalLund = 0, signalJpsi = -1, signalB = -1;

    if( !((process=="BsJPsiPhi")||(process=="BuJPsiK")||(process=="BdJPsiKx")) ) {
        cout<<"!$!#$@$% PROCESS NAME WRONG"<<endl;
        return false;
    }

    if(process=="BsJPsiPhi") signalLund = 531;
    if(process=="BuJPsiK")   signalLund = 521;
    if(process=="BdJPsiKx")  signalLund = 511;

    for( uint i=0; i<genId->size(); ++i ){
        if( abs(genId->at(i)) != signalLund ) continue;
        if(TagMixStatus(i)==2) continue;
        int bCand = i;
        if(abs(genEta->at(bCand))>2.5) continue;
        if(genPt->at(bCand)<10) continue;

        const vector <int>& aD = allDaughters(bCand);
        if(genId->at(aD[0])!=443) continue;
        const vector <int>& aDjpsi = allDaughters(aD[0]);
        if(abs(genId->at(aDjpsi[0]))!=13) continue;
        if(abs(genId->at(aDjpsi[1]))!=13) continue;
        if(abs(genEta->at(aDjpsi[0]))>2.4) continue;
        if(abs(genEta->at(aDjpsi[1]))>2.4) continue;
        if(genPt->at(aDjpsi[0])<3.5) continue;
        if(genPt->at(aDjpsi[1])<3.5) continue;

        signalB = bCand;
        signalJpsi = aD[0];
    }

    if(signalB < 0) return false;

    vector <int> signalMuon;
    vector <int> signalGen;

    const vector <int>& aDjpsi = allDaughters(signalJpsi);
    const vector <int>& aDssb = allDaughters(signalB);
    signalMuon.push_back(aDjpsi[0]);
    signalMuon.push_back(aDjpsi[1]);

    for(auto it:aDjpsi) signalGen.push_back(it);

    if(process=="BsJPsiPhi" || process=="BdJPsiKx"){
        const vector <int>& aDphi = allDaughters(aDssb[1]);
        for(auto it:aDphi) signalGen.push_back(it);
    }
    if(process=="BuJPsiK") signalGen.push_back(aDssb[1]);

    ++nB;

 // generation information
    vector <int> ListB;
    for( uint i=0; i<genId->size(); ++i )
        if( IsB(i) ) 
            ListB.push_back(i);

    int nMuSelected = 0;

    for ( int iMuon = 0; iMuon < nMuons; ++iMuon ){

        //EXCLUDING SIGNAL
        bool skip = false;
        for(auto it:signalGen){
            if( AreOverlapped(genPt->at(it),genEta->at(it),genPhi->at(it),muoPt->at(iMuon),muoEta->at(iMuon),muoPhi->at(iMuon)) ){
                skip = true;
                break;
            }
        }
        if(skip) continue;

        //GLOBAL REQUIREMENT
        if( !(muoType->at(iMuon) & PDEnumString::global) ) continue;
        int itkmu = muonTrack( iMuon, PDEnumString::muInner );
        if(itkmu < 0) continue;

        //GEN INFO
        int genMuIndex = GetClosestGen( muoEta->at(iMuon), muoPhi->at(iMuon), muoPt->at(iMuon) );
        if( genMuIndex < 0 ) continue;
        if(std::find(signalMuon.begin(), signalMuon.end(), genMuIndex) != signalMuon.end()) continue;  //should not be necessary but

        int muoLund = genId->at(genMuIndex);
        int muoAncestor = GetAncestor( genMuIndex, &ListB );

        //KINEMATICS
        if(muoPt->at(iMuon) < minPtMuon) continue;
        if(abs(muoEta->at(iMuon)) > maxEtaMuon) continue;

        //QUALITY
        if( !(( trkQuality->at(itkmu) >> 2 ) & 1) ) continue; 
        if(!muonPassedPreselection(iMuon)) continue;

        ++nMuSelected;
        if((abs(muoLund) == 13) && (muoAncestor > 0)) ++nReal;
        if(abs(muoLund) != 13) ++nFake;


        //TWRITER FILLING
        (tWriter->ssbPt)->push_back( genPt->at( signalB ) );
        (tWriter->ssbEta)->push_back( genEta->at( signalB ) );
        (tWriter->ssbPhi)->push_back( genPhi->at( signalB ) );

        (tWriter->muoPt)->push_back( muoPt->at( iMuon ) );
        (tWriter->muoEta)->push_back( muoEta->at( iMuon ) );
        (tWriter->muoPhi)->push_back( muoPhi->at(iMuon) );

        (tWriter->muoSegmComp)->push_back( muoSegmComp->at(iMuon) );
        (tWriter->muoChi2LP)->push_back( muoChi2LP->at(iMuon) ); 
        (tWriter->muoChi2LM)->push_back( muoChi2LM->at(iMuon) );
        (tWriter->muoGlbTrackTailProb)->push_back( muoGlbTrackTailProb->at(iMuon) );
        (tWriter->muoGlbKinkFinderLOG)->push_back( muoGlbKinkFinderLOG->at(iMuon) );
        (tWriter->muoIValFrac)->push_back( muoIValFrac->at(iMuon) );
        (tWriter->muoLWH)->push_back( (int)round(muoLWH->at(iMuon)) );
        (tWriter->muoNumMatches)->push_back( muoNumMatches->at(iMuon) );
        (tWriter->muoTrkKink)->push_back( muoTrkKink->at(iMuon) );
        (tWriter->muoTimeAtIpInOutErr)->push_back( muoTimeAtIpInOutErr->at(iMuon) );
        (tWriter->muoInnerChi2)->push_back( muoInnerChi2->at(iMuon) );
        (tWriter->muoOuterChi2)->push_back( muoOuterChi2->at(iMuon) );
        (tWriter->muoTrkRelChi2)->push_back( muoTrkRelChi2->at(iMuon) );
        (tWriter->muoVMuonHitComb)->push_back( (int)round(muoVMuonHitComb->at(iMuon)) );
        (tWriter->muoQprod)->push_back( (int)round(muoQprod->at(iMuon)) );

        (tWriter->muoGlbDeltaEtaPhi)->push_back( muoGlbDeltaEtaPhi->at(iMuon) );
        (tWriter->muoStaRelChi2)->push_back( muoStaRelChi2->at(iMuon) );
        (tWriter->muoTimeAtIpInOut)->push_back( muoTimeAtIpInOut->at(iMuon) );
        (tWriter->muoValPixHits)->push_back( (int)round(muoValPixHits->at(iMuon)) );
        (tWriter->muoNTrkVHits)->push_back( (int)round(muoNTrkVHits->at(iMuon)) );
        (tWriter->muoGNchi2)->push_back( muoGNchi2->at(iMuon) );
        (tWriter->muoVMuHits)->push_back( (int)round(muoVMuHits->at(iMuon)) );

        (tWriter->trkDxy)->push_back( trkDxy->at(itkmu));
        (tWriter->trkDz)->push_back( trkDz->at(itkmu) );
        (tWriter->trkExy)->push_back( trkExy->at(itkmu) );
        (tWriter->trkEz)->push_back( trkEz->at(itkmu) );
        (tWriter->muoPFiso)->push_back( GetMuoPFiso(iMuon) );   

        (tWriter->muoLund)->push_back( muoLund );
        (tWriter->muoAncestor)->push_back( muoAncestor ); 

        (tWriter->muoEvt)->push_back( event_tot );
        (tWriter->muoTrkType)->push_back( trkType->at(itkmu) );

    }

    if(nMuSelected == 0) return false;

    tWriter->fill();

// to skim the N-tuple "uncomment" the following line
//  if ( flag ) fillSkim();

    return true;

}


void PDAnalyzer::endJob() {

    tWriter->close();
    cout<<"nB "<<nB<<endl<<"nReal "<<nReal<<endl<<"nFake "<<nFake<<endl<<100.0*nReal/nB<<endl;
    for(int i=0;i<10;++i) cout<<"["<<i<<"]  "<<counter[i]<<endl;
    return;
}


void PDAnalyzer::save() {
#   if UTIL_USE == FULL
    // explicit saving not necessary for "autoSavedObjects"
    autoSave();
#elif UTIL_USE == BARE
    // explicit save histos when not using the full utility

#endif

    return;
}
