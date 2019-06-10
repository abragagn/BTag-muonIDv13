#ifndef PDSecondNtupleData_h
#define PDSecondNtupleData_h
#include <vector>
#include "NtuTool/Common/interface/TreeWrapper.h"
using namespace std;

class PDSecondNtupleData: public virtual TreeWrapper {

public:

void Reset()    { autoReset(); }

PDSecondNtupleData() {

    ssbPt               = new vector <float>; 
    ssbEta              = new vector <float>; 
    ssbPhi              = new vector <float>; 

    muoPt               = new vector <float>; 
    muoEta              = new vector <float>;
    muoPhi              = new vector <float>;
    trkDxy              = new vector <float>;
    trkExy              = new vector <float>;
    trkDz               = new vector <float>;
    trkEz               = new vector <float>;

    muoNumMatches       = new vector <float>;
    muoChi2LP           = new vector <float>;
    muoTrkKink          = new vector <float>;
    muoSegmComp         = new vector <float>;
    muoChi2LM           = new vector <float>;
    muoTrkRelChi2       = new vector <float>;
    muoGlbTrackTailProb = new vector <float>;
    muoGlbKinkFinderLOG = new vector <float>;
    muoGlbDeltaEtaPhi   = new vector <float>;
    muoStaRelChi2       = new vector <float>;
    muoTimeAtIpInOut    = new vector <float>;
    muoTimeAtIpInOutErr = new vector <float>;
    muoInnerChi2        = new vector <float>;
    muoIValFrac         = new vector <float>;    
    muoValPixHits       = new vector <int>;
    muoNTrkVHits        = new vector <int>;
    muoLWH              = new vector <int>;
    muoOuterChi2        = new vector <float>;
    muoGNchi2           = new vector <float>;
    muoVMuHits          = new vector <int>;
    muoVMuonHitComb     = new vector <int>;
    muoQprod            = new vector <int>;
    
    muoPFiso            = new vector <float>;

    muoLund             = new vector <int>;
    muoAncestor         = new vector <int>;

    muoEvt              = new vector <int>;
    muoTrkType          = new vector <int>;


}
virtual ~PDSecondNtupleData() {
}

void initTree() {
    treeName = "PDsecondTree";
    
    setBranch( "ssbPt", &ssbPt , 8192, 99, &b_ssbPt );
    setBranch( "ssbEta", &ssbEta , 8192, 99, &b_ssbEta );
    setBranch( "ssbPhi", &ssbPhi , 8192, 99, &b_ssbPhi );

    setBranch( "muoPt", &muoPt , 8192, 99, &b_muoPt );
    setBranch( "muoEta", &muoEta , 8192, 99, &b_muoEta );
    setBranch( "muoPhi", &muoPhi , 8192, 99, &b_muoPhi );
    setBranch( "trkDxy", &trkDxy , 8192, 99, &b_trkDxy );
    setBranch( "trkExy", &trkExy , 8192, 99, &b_trkExy );
    setBranch( "trkDz", &trkDz , 8192, 99, &b_trkDz );
    setBranch( "trkEz", &trkEz , 8192, 99, &b_trkEz );

    setBranch( "muoNumMatches", &muoNumMatches , 8192, 99, &b_muoNumMatches );
    setBranch( "muoChi2LP", &muoChi2LP , 8192, 99, &b_muoChi2LP );
    setBranch( "muoTrkKink", &muoTrkKink , 8192, 99, &b_muoTrkKink );
    setBranch( "muoSegmComp", &muoSegmComp , 8192, 99, &b_muoSegmComp );
    setBranch( "muoChi2LM", &muoChi2LM , 8192, 99, &b_muoChi2LM );
    setBranch( "muoTrkRelChi2", &muoTrkRelChi2 , 8192, 99, &b_muoTrkRelChi2 );
    setBranch( "muoGlbTrackTailProb", &muoGlbTrackTailProb , 8192, 99, &b_muoGlbTrackTailProb );
    setBranch( "muoGlbKinkFinderLOG", &muoGlbKinkFinderLOG , 8192, 99, &b_muoGlbKinkFinderLOG );
    setBranch( "muoGlbDeltaEtaPhi", &muoGlbDeltaEtaPhi , 8192, 99, &b_muoGlbDeltaEtaPhi );
    setBranch( "muoStaRelChi2", &muoStaRelChi2 , 8192, 99, &b_muoStaRelChi2 );
    setBranch( "muoTimeAtIpInOut", &muoTimeAtIpInOut , 8192, 99, &b_muoTimeAtIpInOut );
    setBranch( "muoTimeAtIpInOutErr", &muoTimeAtIpInOutErr , 8192, 99, &b_muoTimeAtIpInOutErr );
    setBranch( "muoInnerChi2", &muoInnerChi2 , 8192, 99, &b_muoInnerChi2 );
    setBranch( "muoIValFrac", &muoIValFrac , 8192, 99, &b_muoIValFrac );
    setBranch( "muoValPixHits", &muoValPixHits , 8192, 99, &b_muoValPixHits );
    setBranch( "muoNTrkVHits", &muoNTrkVHits , 8192, 99, &b_muoNTrkVHits );
    setBranch( "muoLWH", &muoLWH , 8192, 99, &b_muoLWH );
    setBranch( "muoOuterChi2", &muoOuterChi2 , 8192, 99, &b_muoOuterChi2 );
    setBranch( "muoGNchi2", &muoGNchi2 , 8192, 99, &b_muoGNchi2 );
    setBranch( "muoVMuHits", &muoVMuHits , 8192, 99, &b_muoVMuHits );
    setBranch( "muoVMuonHitComb", &muoVMuonHitComb , 8192, 99, &b_muoVMuonHitComb );
    setBranch( "muoQprod", &muoQprod , 8192, 99, &b_muoQprod );

    setBranch( "muoPFiso", &muoPFiso , 8192, 99, &b_muoPFiso );

    setBranch( "muoLund", &muoLund , 8192, 99, &b_muoLund );
    setBranch( "muoAncestor", &muoAncestor , 8192, 99, &b_muoAncestor );

    setBranch( "muoEvt", &muoEvt , 8192, 99, &b_muoEvt );
    setBranch( "muoTrkType", &muoTrkType , 8192, 99, &b_muoTrkType );

}
vector <float> *ssbPt, *ssbEta, *ssbPhi;

vector <int> *muoValPixHits, *muoNTrkVHits, *muoLWH, *muoVMuHits, *muoVMuonHitComb, *muoQprod;
vector <int> *muoLund, *muoAncestor, *muoEvt, *muoTrkType;
vector <float> *muoPt, *muoEta, *muoPhi, *trkDxy, *trkExy, *trkDz, *trkEz;
vector <float> *muoNumMatches, *muoChi2LP, *muoTrkKink, *muoSegmComp, *muoChi2LM, *muoTrkRelChi2;
vector <float> *muoGlbTrackTailProb, *muoGlbKinkFinderLOG, *muoGlbDeltaEtaPhi, *muoStaRelChi2;
vector <float> *muoTimeAtIpInOut, *muoTimeAtIpInOutErr, *muoInnerChi2, *muoIValFrac, *muoGNchi2, *muoOuterChi2;
vector <float> *muoPFiso;

TBranch *b_ssbPt, *b_ssbEta, *b_ssbPhi;
TBranch *b_muoPt, *b_muoEta, *b_muoPhi, *b_trkDxy, *b_trkExy, *b_trkDz, *b_trkEz;
TBranch *b_muoLund, *b_muoAncestor, *b_muoEvt, *b_muoTrkType;
TBranch *b_muoValPixHits, *b_muoNTrkVHits, *b_muoLWH, *b_muoVMuHits, *b_muoVMuonHitComb, *b_muoQprod;
TBranch *b_muoNumMatches, *b_muoChi2LP, *b_muoTrkKink, *b_muoSegmComp, *b_muoChi2LM, *b_muoTrkRelChi2;
TBranch *b_muoGlbTrackTailProb, *b_muoGlbKinkFinderLOG, *b_muoGlbDeltaEtaPhi, *b_muoStaRelChi2;
TBranch *b_muoTimeAtIpInOut, *b_muoTimeAtIpInOutErr, *b_muoInnerChi2, *b_muoIValFrac, *b_muoGNchi2, *b_muoOuterChi2;
TBranch *b_muoPFiso;

 private:

PDSecondNtupleData ( const PDSecondNtupleData& a );
PDSecondNtupleData& operator=( const PDSecondNtupleData& a );

};

#endif

