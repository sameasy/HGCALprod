///Removed EventSetup use from RecHitTools class.

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "DataFormats/Math/interface/deltaPhi.h"
// track data formats
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/CaloGeometryTools/interface/Transform3DPJ.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TH1F.h"
#include "TTree.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
# include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
# include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"

# include "RecoEgamma/EgammaTools/interface/HGCalClusterTools.h"
# include "RecoEgamma/EgammaTools/interface/HGCalShowerShapeHelper.h"
# include "DataFormats/EgammaCandidates/interface/Photon.h" 


#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;

  // HGC_helpers

class HGCalEGMAnalysis : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
 public:
  //
  // constructors and destructor
  //
  typedef ROOT::Math::Transform3DPJ::Point Point;

  // approximative geometrical values
  static constexpr float hgcalOuterRadius_ = 160.;
  static constexpr float hgcalInnerRadius_ = 25.;

  HGCalEGMAnalysis();
  explicit HGCalEGMAnalysis(const edm::ParameterSet &);
  ~HGCalEGMAnalysis();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  virtual void beginRun(edm::Run const &iEvent, edm::EventSetup const &) override;
  virtual void endRun(edm::Run const &iEvent, edm::EventSetup const &) override;
  DetId simToReco(const HGCalGeometry* geom, unsigned int simId);
 private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;


  void clearVariables();

  void retrieveLayerPositions(const edm::EventSetup &, unsigned layers);

  // ---------parameters ----------------------------
  std::string inputTag_Reco_;
  //  std::string inputTag_ReReco_;
  HGCalClusterTools algo_HoverE_;
  HGCalShowerShapeHelper showerShapeHelper_;
  // ----------member data ---------------------------


  edm::EDGetTokenT<reco::CaloClusterCollection> clusters_;
  edm::EDGetTokenT<std::vector<SimVertex>> simVertices_;
  edm::EDGetTokenT<edm::HepMCProduct> hev_;

  edm::EDGetTokenT<std::vector<reco::PFRecHit>> pfrechit_;
  edm::EDGetTokenT<std::vector<reco::GsfElectron>> electrons_;
  edm::EDGetTokenT<std::vector<reco::Photon>> photons_;
  edm::EDGetTokenT<std::vector<reco::GenParticle> > genParticles_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom;
  //  const edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> pdtToken_; //added
  const edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> pdtToken_; //added
  //  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  //  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom2_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;


  TTree *t_;

  ////////////////////
  // event
  //
  edm::RunNumber_t ev_run_;
  edm::LuminosityBlockNumber_t ev_lumi_;
  edm::EventNumber_t ev_event_;
  float vtx_x_;
  float vtx_y_;
  float vtx_z_;


  ////////////////////
  // reco::GenParticles
  //
  std::vector<float> gen_eta_;
  std::vector<float> gen_phi_;
  std::vector<float> gen_pt_;
  std::vector<float> gen_energy_;
  std::vector<int> gen_charge_;
  std::vector<int> gen_pdgid_;
  std::vector<int> gen_status_;
  std::vector<std::vector<int>> gen_daughters_;


  ////////////////////
  // Ecal Driven GsfElectrons From MultiClusters
  //

  std::vector<float> ele_charge_;
  std::vector<float> ele_eta_;
  std::vector<float> ele_phi_;
  std::vector<float> ele_pt_;
  std::vector<ROOT::Math::XYZPoint> ele_scpos_;
  std::vector<float> ele_sceta_;
  std::vector<float> ele_scphi_;
  //  std::vector<uint32_t> ele_seedlayer_;
  std::vector<ROOT::Math::XYZPoint> ele_seedpos_;
  std::vector<float> ele_seedeta_;
  std::vector<float> ele_seedphi_;
  std::vector<float> ele_seedenergy_;
  std::vector<float> ele_energy_;
  std::vector<float> ele_rawenergy_;
  std::vector<float> ele_hoe_;
  std::vector<float> ele_rvar_;


  std::vector<float> pho_charge_;
  std::vector<float> pho_eta_;
  std::vector<float> pho_phi_;
  std::vector<float> pho_pt_;
  std::vector<ROOT::Math::XYZPoint> pho_scpos_;
  std::vector<float> pho_sceta_;
  std::vector<float> pho_scphi_;
  //  std::vector<uint32_t> pho_seedlayer_;
  std::vector<ROOT::Math::XYZPoint> pho_seedpos_;
  std::vector<float> pho_seedeta_;
  std::vector<float> pho_seedphi_;
  std::vector<float> pho_seedenergy_;
  std::vector<float> pho_energy_;
  std::vector<float> pho_rawenergy_;
  std::vector<float> pho_hoe_;
  std::vector<float> pho_rvar_;

  /*
  std::vector<float> ele_energyEE_;
  std::vector<float> ele_energyFH_;
  std::vector<float> ele_energyBH_;
  std::vector<float> ele_isEB_;
  std::vector<float> ele_hoe_;
  std::vector<float> ele_numClinSC_;
  std::vector<float> ele_track_dxy_;
  std::vector<float> ele_track_dz_;
  std::vector<float> ele_track_simdxy_;
  std::vector<float> ele_track_simdz_;
  std::vector<float> ele_deltaEtaSuperClusterTrackAtVtx_;
  std::vector<float> ele_deltaPhiSuperClusterTrackAtVtx_;
  std::vector<float> ele_deltaEtaEleClusterTrackAtCalo_;
  std::vector<float> ele_deltaPhiEleClusterTrackAtCalo_;
  std::vector<float> ele_deltaEtaSeedClusterTrackAtCalo_;
  std::vector<float> ele_deltaPhiSeedClusterTrackAtCalo_;
  std::vector<float> ele_eSuperClusterOverP_;
  std::vector<float> ele_eSeedClusterOverP_;
  std::vector<float> ele_eSeedClusterOverPout_;
  std::vector<float> ele_eEleClusterOverPout_;
  std::vector<std::vector<uint32_t>>
      ele_pfClusterIndex_;  // the second index runs through the corresponding
                                         // PFClustersHGCalFromMultiClusters

  */

  ////////////////////
  // helper classes
  //
  float vz_;  // primary vertex z position
  // to keep track of the RecHits stored within the cluster loops
  std::set<DetId> storedRecHits_;
  int algo_;
  HGCalDepthPreClusterer pre_;
  //  hgcal::RecHitTools recHitTools_;
  hgcal::RecHitTools recHitTools;
  // -------convenient tool to deal with simulated tracks
  FSimEvent *mySimEvent_;
  edm::ParameterSet particleFilter_;
  std::vector<float> layerPositions_;


  // and also the magnetic field
  MagneticField const *aField_;

};

HGCalEGMAnalysis::HGCalEGMAnalysis() { ; }

HGCalEGMAnalysis::HGCalEGMAnalysis(const edm::ParameterSet &iConfig)
  :   inputTag_Reco_(iConfig.getParameter<std::string>("inputTag_Reco")),
      //      inputTag_ReReco_(iConfig.getParameter<std::string>("inputTag_ReReco")),
      showerShapeHelper_(consumesCollector()),
      tok_geom{esConsumes<CaloGeometry, CaloGeometryRecord>()},
      pdtToken_(esConsumes<edm::Transition::BeginRun>()),
      //      tok_geom_(esConsumes<edm::Transition::BeginRun>()),
//      tok_geom2_(esConsumes<CaloGeometry, CaloGeometryRecord>()),

      tok_magField_(esConsumes<edm::Transition::BeginRun>()),
    
      particleFilter_(iConfig.getParameter<edm::ParameterSet>("TestParticleFilter")) {
  // now do what ever initialization is needed
  mySimEvent_ = new FSimEvent(particleFilter_);


  clusters_ = consumes<reco::CaloClusterCollection>(edm::InputTag("hgcalLayerClusters"));
  hev_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));

  

  genParticles_ = consumes<std::vector<reco::GenParticle>>(edm::InputTag("genParticles"));

  electrons_ = consumes<std::vector<reco::GsfElectron>>(edm::InputTag("ecalDrivenGsfElectronsHGC","",inputTag_Reco_));
  photons_ = consumes<std::vector<reco::Photon>>(edm::InputTag("photonsHGC","",inputTag_Reco_));

  //  pfrechit_ = consumes <std::vector <reco::PFRecHit> >(iConfig.getParameter <edm::InputTag>("PFRecHits"));
  pfrechit_ = consumes <std::vector <reco::PFRecHit> >(edm::InputTag("particleFlowRecHitHGC"));

  vertices_ = consumes<std::vector<reco::Vertex>>(edm::InputTag("offlinePrimaryVertices"));




  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  fs->make<TH1F>("total", "total", 100, 0, 5.);

  t_ = fs->make<TTree>("hgc", "hgc");

  // event info
  t_->Branch("event", &ev_event_);
  t_->Branch("lumi", &ev_lumi_);
  t_->Branch("run", &ev_run_);
  t_->Branch("vtx_x", &vtx_x_);
  t_->Branch("vtx_y", &vtx_y_);
  t_->Branch("vtx_z", &vtx_z_);



  t_->Branch("gen_eta", &gen_eta_);
  t_->Branch("gen_phi", &gen_phi_);
  t_->Branch("gen_pt", &gen_pt_);
  t_->Branch("gen_energy", &gen_energy_);
  t_->Branch("gen_charge", &gen_charge_);
  t_->Branch("gen_pdgid", &gen_pdgid_);
  t_->Branch("gen_status", &gen_status_);
  t_->Branch("gen_daughters", &gen_daughters_);


  ////////////////////
  // Ecal Driven Gsf Electrons From MultiClusters
  //
  
  t_->Branch("ele_charge", &ele_charge_);
  t_->Branch("ele_eta", &ele_eta_);
  t_->Branch("ele_phi", &ele_phi_);
  t_->Branch("ele_pt", &ele_pt_);
  t_->Branch("ele_scpos", &ele_scpos_);
  t_->Branch("ele_sceta", &ele_sceta_);
  t_->Branch("ele_scphi", &ele_scphi_);
  //  t_->Branch("ele_seedlayer", &ele_seedlayer_);
  t_->Branch("ele_seedpos", &ele_seedpos_);
  t_->Branch("ele_seedeta", &ele_seedeta_);
  t_->Branch("ele_seedphi", &ele_seedphi_);
  t_->Branch("ele_seedenergy", &ele_seedenergy_);
  t_->Branch("ele_energy", &ele_energy_);
  t_->Branch("ele_rawenergy", &ele_rawenergy_);
  t_->Branch("ele_hoe", &ele_hoe_);
  t_->Branch("ele_rvar", &ele_rvar_);

  t_->Branch("pho_charge", &pho_charge_);
  t_->Branch("pho_eta", &pho_eta_);
  t_->Branch("pho_phi", &pho_phi_);
  t_->Branch("pho_pt", &pho_pt_);
  t_->Branch("pho_scpos", &pho_scpos_);
  t_->Branch("pho_sceta", &pho_sceta_);
  t_->Branch("pho_scphi", &pho_scphi_);
  //  t_->Branch("pho_seedlayer", &pho_seedlayer_);
  t_->Branch("pho_seedpos", &pho_seedpos_);
  t_->Branch("pho_seedeta", &pho_seedeta_);
  t_->Branch("pho_seedphi", &pho_seedphi_);
  t_->Branch("pho_seedenergy", &pho_seedenergy_);
  t_->Branch("pho_energy", &pho_energy_);
  t_->Branch("pho_rawenergy", &pho_rawenergy_);
  t_->Branch("pho_hoe", &pho_hoe_);
  t_->Branch("pho_rvar", &pho_rvar_);

  /*
  t_->Branch("ele_energyEE", &ele_energyEE_);
  t_->Branch("ele_energyFH", &ele_energyFH_);
  t_->Branch("ele_energyBH", &ele_energyBH_);
  t_->Branch("ele_isEB", &ele_isEB_);
  t_->Branch("ele_hoe", &ele_hoe_);
  t_->Branch("ele_numClinSC", &ele_numClinSC_);
  t_->Branch("ele_track_dxy", &ele_track_dxy_);
  t_->Branch("ele_track_dz", &ele_track_dz_);
  t_->Branch("ele_track_simdxy", &ele_track_simdxy_);
  t_->Branch("ele_track_simdz", &ele_track_simdz_);
  t_->Branch("ele_deltaEtaSuperClusterTrackAtVtx",
	     &ele_deltaEtaSuperClusterTrackAtVtx_);
  t_->Branch("ele_deltaPhiSuperClusterTrackAtVtx",
	     &ele_deltaPhiSuperClusterTrackAtVtx_);
  t_->Branch("ele_deltaEtaEleClusterTrackAtCalo",
	     &ele_deltaEtaEleClusterTrackAtCalo_);
  t_->Branch("ele_deltaPhiEleClusterTrackAtCalo",
	     &ele_deltaPhiEleClusterTrackAtCalo_);
  t_->Branch("ele_deltaEtaSeedClusterTrackAtCalo",
	     &ele_deltaEtaSeedClusterTrackAtCalo_);
  t_->Branch("ele_deltaPhiSeedClusterTrackAtCalo",
	     &ele_deltaPhiSeedClusterTrackAtCalo_);
  t_->Branch("ele_eSuperClusterOverP", &ele_eSuperClusterOverP_);
  t_->Branch("ele_eSeedClusterOverP", &ele_eSeedClusterOverP_);
  t_->Branch("ele_eSeedClusterOverPout", &ele_eSeedClusterOverPout_);
  t_->Branch("ele_eEleClusterOverPout", &ele_eEleClusterOverPout_);
  t_->Branch("ele_pfClusterIndex", &ele_pfClusterIndex_);
  */

  ////////////////////

}
HGCalEGMAnalysis::~HGCalEGMAnalysis() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}




void HGCalEGMAnalysis::clearVariables() {
  ev_run_ = 0;
  ev_lumi_ = 0;
  ev_event_ = 0;
  vtx_x_ = 0;
  vtx_y_ = 0;
  vtx_z_ = 0;

  ////////////////////
  // GenParticles
  ////////////////////
  // reco::GenParticles
  //
  gen_eta_.clear();
  gen_phi_.clear();
  gen_pt_.clear();
  gen_energy_.clear();
  gen_charge_.clear();
  gen_pdgid_.clear();
  gen_status_.clear();
  gen_daughters_.clear();

  ////////////////////
  //  Ecal Driven Gsf Electrons From MultiClusters
  //
  ele_charge_.clear();
  ele_eta_.clear();
  ele_phi_.clear();
  ele_pt_.clear();
  ele_scpos_.clear();
  ele_sceta_.clear();
  ele_scphi_.clear();
  //  ele_seedlayer_.clear();
  ele_seedpos_.clear();
  ele_seedeta_.clear();
  ele_seedphi_.clear();
  ele_seedenergy_.clear();
  ele_energy_.clear();
  ele_hoe_.clear();
  ele_rvar_.clear();
  /*
  ele_energyEE_.clear();
  ele_energyFH_.clear();
  ele_energyBH_.clear();
  ele_isEB_.clear();
  ele_hoe_.clear();
  ele_numClinSC_.clear();
  ele_track_dxy_.clear();
  ele_track_dz_.clear();
  ele_track_simdxy_.clear();
  ele_track_simdz_.clear();
  ele_deltaEtaSuperClusterTrackAtVtx_.clear();
  ele_deltaPhiSuperClusterTrackAtVtx_.clear();
  ele_deltaEtaEleClusterTrackAtCalo_.clear();
  ele_deltaPhiEleClusterTrackAtCalo_.clear();
  ele_deltaEtaSeedClusterTrackAtCalo_.clear();
  ele_deltaPhiSeedClusterTrackAtCalo_.clear();
  ele_eSuperClusterOverP_.clear();
  ele_eSeedClusterOverP_.clear();
  ele_eSeedClusterOverPout_.clear();
  ele_eEleClusterOverPout_.clear();
  ele_pfClusterIndex_.clear();
  */
  ////////////////////
  pho_charge_.clear();
  pho_eta_.clear();
  pho_phi_.clear();
  pho_pt_.clear();
  pho_scpos_.clear();
  pho_sceta_.clear();
  pho_scphi_.clear();
  //  pho_seedlayer_.clear();
  pho_seedpos_.clear();
  pho_seedeta_.clear();
  pho_seedphi_.clear();
  pho_seedenergy_.clear();
  pho_energy_.clear();
  pho_hoe_.clear();
  pho_rvar_.clear();

}

void HGCalEGMAnalysis::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  clearVariables();

  //  edm::ESHandle<CaloGeometry> geom;
  //  iSetup.get<CaloGeometryRecord>().get(geom);
  //  recHitTools.setGeometry(*(geom.product()));

  const CaloGeometry *geom = &iSetup.getData(tok_geom);
  recHitTools.setGeometry(*geom);

  Handle<reco::CaloClusterCollection> clusterHandle;
  iEvent.getByToken(clusters_, clusterHandle);

  Handle<edm::HepMCProduct> hevH;
  iEvent.getByToken(hev_, hevH);


  Handle<std::vector<reco::Vertex>> verticesHandle;
  iEvent.getByToken(vertices_, verticesHandle);
  auto const &vertices = *verticesHandle;



  HepMC::GenVertex *primaryVertex = *(hevH)->GetEvent()->vertices_begin();
  float vx_ = primaryVertex->position().x() / 10.;  // to put in official units
  float vy_ = primaryVertex->position().y() / 10.;
  vz_ = primaryVertex->position().z() / 10.;
  Point sim_pv(vx_, vy_, vz_);





  
  
  Handle<std::vector<reco::GenParticle>> genParticlesHandle;
  iEvent.getByToken(genParticles_, genParticlesHandle);
  //  std::cout<<"looping over genparts ....."<<std::endl;
  for (std::vector<reco::GenParticle>::const_iterator it_p = genParticlesHandle->begin();
       it_p != genParticlesHandle->end(); ++it_p) {


    gen_eta_.push_back(it_p->eta());
    gen_phi_.push_back(it_p->phi());
    gen_pt_.push_back(it_p->pt());
    gen_energy_.push_back(it_p->energy());
    gen_charge_.push_back(it_p->charge());
    gen_pdgid_.push_back(it_p->pdgId());
    gen_status_.push_back(it_p->status());

    //    std::cout<<"genpart pdgid,eta,phi,en =====> "<<it_p->pdgId()<<" ,"<<it_p->eta()<<" ,"<<it_p->phi()<<" ,"<<it_p->energy()<<std::endl;
    std::vector<int> daughters(it_p->daughterRefVector().size(), 0);
    for (unsigned j = 0; j < it_p->daughterRefVector().size(); ++j) {
      daughters[j] = static_cast<int>(it_p->daughterRefVector().at(j).key());
    }
    gen_daughters_.push_back(daughters);
  }
  


  edm::Handle <std::vector <reco::PFRecHit> > h_PFRecHit;
  iEvent.getByToken(pfrechit_, h_PFRecHit);
  auto recHits = *h_PFRecHit;
  showerShapeHelper_.initPerEvent(iSetup, recHits);
  
  Handle<std::vector<reco::GsfElectron>> eleHandle;
  iEvent.getByToken(electrons_, eleHandle);
  const std::vector<reco::GsfElectron> &electrons = *eleHandle;
  
  auto layerClusters = *clusterHandle;

  //  std::cout<<"looping over electronss ....."<<std::endl;
  for (auto const &ele : electrons) {
    std::vector<uint32_t> pfclustersIndex;
    auto const &sc = ele.superCluster();
    //float hoe = 0.;

    double hoe = algo_HoverE_.hadEnergyInCone(
					      ele.superCluster()->eta(),
					      ele.superCluster()->phi(),
					      layerClusters,
					      0.0,
					      0.15,
					      0.0,
					      0.0,
					      HGCalClusterTools::EType::ENERGY
					      );
    
    hoe /= ele.superCluster()->energy();
    
    
    
    auto ssCalc = showerShapeHelper_.createCalc(
						*ele.superCluster(),
						0.0,
						0.0
						);
    
    double Rvar = ssCalc.getRvar(2.8);
    
    //    std::cout<<"electron charge,eta,phi,energy,hoe,rvar =====> "<<ele.charge()<<" ,"<<ele.eta()<<" ,"<<ele.phi()<<" ,"<<ele.energy()<<" ,"<<hoe<<" ,"<<Rvar<<std::endl;
    //float energyEE = 0.;
        //float energyFH = 0.;
        //float energyBH = 0.;
    ele_charge_.push_back(ele.charge());
    ele_eta_.push_back(ele.eta());
    ele_phi_.push_back(ele.phi());
    ele_pt_.push_back(ele.pt());
    ele_scpos_.push_back(ele.superCluster()->position());
    ele_sceta_.push_back(ele.superCluster()->eta());
    ele_scphi_.push_back(ele.superCluster()->phi());
    //        ele_seedlayer_.push_back(
    //            recHitTools_.getLayerWithOffset(ele.superCluster()->seed()->seed()));
    ele_seedpos_.push_back(ele.superCluster()->seed()->position());
    ele_seedeta_.push_back(ele.superCluster()->seed()->eta());
    ele_seedphi_.push_back(ele.superCluster()->seed()->phi());
    ele_seedenergy_.push_back(ele.superCluster()->seed()->energy());
    ele_energy_.push_back(ele.energy());
    ele_rawenergy_.push_back(ele.superCluster()->rawEnergy());
    ele_hoe_.push_back(hoe);
    ele_rvar_.push_back(Rvar);
    /*
      ele_energyEE_.push_back(energyEE);
      ele_energyFH_.push_back(energyFH);
      ele_energyBH_.push_back(energyBH);
      ele_isEB_.push_back(ele.isEB());
      ele_hoe_.push_back(hoe);
      ele_numClinSC_.push_back(sc->clusters().size());
      ele_track_dxy_.push_back(ele.gsfTrack()->dxy(vertices[0].position()));
      ele_track_dz_.push_back(ele.gsfTrack()->dz(vertices[0].position()));
      ele_track_simdxy_.push_back(ele.gsfTrack()->dxy(sim_pv));
      ele_track_simdz_.push_back(ele.gsfTrack()->dz(sim_pv));
      ele_deltaEtaSuperClusterTrackAtVtx_.push_back(
      ele.deltaEtaSuperClusterTrackAtVtx());
      ele_deltaPhiSuperClusterTrackAtVtx_.push_back(
      ele.deltaPhiSuperClusterTrackAtVtx());
      ele_deltaEtaEleClusterTrackAtCalo_.push_back(ele.deltaEtaEleClusterTrackAtCalo());
      ele_deltaPhiEleClusterTrackAtCalo_.push_back(ele.deltaPhiEleClusterTrackAtCalo());
      ele_deltaEtaSeedClusterTrackAtCalo_.push_back(
      ele.deltaEtaSeedClusterTrackAtCalo());
      ele_deltaPhiSeedClusterTrackAtCalo_.push_back(
      ele.deltaPhiSeedClusterTrackAtCalo());
      ele_eSuperClusterOverP_.push_back(ele.eSuperClusterOverP());
      ele_eSeedClusterOverP_.push_back(ele.eSeedClusterOverP());
      ele_eSeedClusterOverPout_.push_back(ele.eSeedClusterOverPout());
      ele_eEleClusterOverPout_.push_back(ele.eEleClusterOverPout());
      ele_pfClusterIndex_.push_back(pfclustersIndex);
    */
  }  // End of loop over electrons


  Handle<std::vector<reco::Photon>> phoHandle;
  iEvent.getByToken(photons_, phoHandle);
  const std::vector<reco::Photon> &photons = *phoHandle;
  
  //  std::cout<<"looping over photons ....."<<std::endl;
  for (auto const &pho : photons) {
    
    auto const &psc = pho.superCluster();
    //float phoe = 0.;
    
    double hoe = algo_HoverE_.hadEnergyInCone(
					      pho.superCluster()->eta(),
					      pho.superCluster()->phi(),
					      layerClusters,
					      0.0,
					      0.15,
					      0.0,
					      0.0,
					      HGCalClusterTools::EType::ENERGY
					      );
    
    hoe /= pho.superCluster()->energy();
    
    auto ssCalc = showerShapeHelper_.createCalc(
						*pho.superCluster(),
						0.0,
						0.0
						);
    
    double Rvar = ssCalc.getRvar(2.8);

    //    std::cout<<"photon eta,phi,energy,hoe,rvar =====> "<<" ,"<<pho.eta()<<" ,"<<pho.phi()<<" ,"<<pho.energy()<<" ,"<<hoe<<" ,"<<Rvar<<std::endl;
    //float energyEE = 0.;
    //float energyFH = 0.;
    //float energyBH = 0.;
    //ecalDrivenGsfele_charge_.push_back(ele.charge());
    pho_eta_.push_back(pho.eta());
    pho_phi_.push_back(pho.phi());
    pho_pt_.push_back(pho.pt());
    pho_scpos_.push_back(pho.superCluster()->position());
    pho_sceta_.push_back(pho.superCluster()->eta());
    pho_scphi_.push_back(pho.superCluster()->phi());
    //        pho_seedlayer_.push_back(
    //            recHitTools_.getLayerWithOffset(pho.superCluster()->seed()->seed()));
    pho_seedpos_.push_back(pho.superCluster()->seed()->position());
    pho_seedeta_.push_back(pho.superCluster()->seed()->eta());
    pho_seedphi_.push_back(pho.superCluster()->seed()->phi());
    pho_seedenergy_.push_back(pho.superCluster()->seed()->energy());
    pho_energy_.push_back(pho.energy());
    pho_rawenergy_.push_back(pho.superCluster()->rawEnergy());
    pho_hoe_.push_back(hoe);
    pho_rvar_.push_back(Rvar);
    /*
      ele_energyEE_.push_back(energyEE);
      ele_energyFH_.push_back(energyFH);
      ele_energyBH_.push_back(energyBH);
      ele_isEB_.push_back(ele.isEB());
      ele_hoe_.push_back(hoe);
      ele_numClinSC_.push_back(sc->clusters().size());
      ele_track_dxy_.push_back(ele.gsfTrack()->dxy(vertices[0].position()));
      ele_track_dz_.push_back(ele.gsfTrack()->dz(vertices[0].position()));
      ele_track_simdxy_.push_back(ele.gsfTrack()->dxy(sim_pv));
      ele_track_simdz_.push_back(ele.gsfTrack()->dz(sim_pv));
      ele_deltaEtaSuperClusterTrackAtVtx_.push_back(
      ele.deltaEtaSuperClusterTrackAtVtx());
      ele_deltaPhiSuperClusterTrackAtVtx_.push_back(
      ele.deltaPhiSuperClusterTrackAtVtx());
      ele_deltaEtaEleClusterTrackAtCalo_.push_back(ele.deltaEtaEleClusterTrackAtCalo());
      ele_deltaPhiEleClusterTrackAtCalo_.push_back(ele.deltaPhiEleClusterTrackAtCalo());
      ele_deltaEtaSeedClusterTrackAtCalo_.push_back(
      ele.deltaEtaSeedClusterTrackAtCalo());
      ele_deltaPhiSeedClusterTrackAtCalo_.push_back(
      ele.deltaPhiSeedClusterTrackAtCalo());
      ele_eSuperClusterOverP_.push_back(ele.eSuperClusterOverP());
      ele_eSeedClusterOverP_.push_back(ele.eSeedClusterOverP());
      ele_eSeedClusterOverPout_.push_back(ele.eSeedClusterOverPout());
      ele_eEleClusterOverPout_.push_back(ele.eEleClusterOverPout());
      ele_pfClusterIndex_.push_back(pfclustersIndex);
    */
  }  // End of loop over photons
  
  
  
  
  
  ev_event_ = iEvent.id().event();
  ev_lumi_ = iEvent.id().luminosityBlock();
  ev_run_ = iEvent.id().run();
  
  vtx_x_ = vx_;
  vtx_y_ = vy_;
  vtx_z_ = vz_;
  
  t_->Fill();
}


void HGCalEGMAnalysis::beginRun(edm::Run const &iEvent, edm::EventSetup const &es) {
  //  edm::ESHandle<HepPDT::ParticleDataTable> pdt;
  //  es.getData(pdtToken_);
  //pdt = es.get(pdtToken_);
  //  auto const pdt = es.getData(pdtToken_);
  //  mySimEvent_->initializePdt(&(*pdt));

  const HepPDT::ParticleDataTable* pdt = &es.getData(pdtToken_);

  mySimEvent_->initializePdt(pdt);


  //recHitTools_.getEventSetup(es);

  //  edm::ESHandle<CaloGeometry> geom;
  //  es.get<CaloGeometryRecord>().get(geom);
  //  const CaloGeometry* geom = &es.getData(tok_geom_);


  //const CaloGeometry& geom = *(es.getHandle(tok_geom_));


  //recHitTools_.setGeometry(geom);


  //retrieveLayerPositions(es, recHitTools_.lastLayerBH());

  //  edm::ESHandle<MagneticField> magfield;
  //  es.get<IdealMagneticFieldRecord>().get(magfield);
  const MagneticField* magfield = &es.getData(tok_magField_);

  aField_ = magfield;
  //es.get<CaloGeometryRecord>().get(geometry);
}

void HGCalEGMAnalysis::endRun(edm::Run const &iEvent, edm::EventSetup const &) {}

void HGCalEGMAnalysis::beginJob() { ; }

// ------------ method called once each job just after ending the event loop
// ------------
void HGCalEGMAnalysis::endJob() {}

// ------------ method to be called once
// --------------------------------------------------

//void HGCalEGMAnalysis::retrieveLayerPositions(const edm::EventSetup &es, unsigned layers) {
  //  recHitTools_.getEventSetup(es);

  //  edm::ESHandle<CaloGeometry> geom;
  //  es.get<CaloGeometryRecord>().get(geom);
  //  recHitTools_.setGeometry(*geom);
  /*
    const CaloGeometry& geom2 = *(es.getHandle(tok_geom_));
    recHitTools_.setGeometry(geom2);
    
    
    
    for (unsigned ilayer = 1; ilayer <= layers; ++ilayer) {
    const GlobalPoint pos = recHitTools_.getPositionLayer(ilayer);
    layerPositions_.push_back(pos.z());
    }
  */

//}



// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------

void HGCalEGMAnalysis::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation
  // Please change this to state exactly what you do use, even if it is no
  // parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

/*
   Surface::RotationType HGCalEGMAnalysis::rotation( const GlobalVector& zDir)
   const
   {
   GlobalVector zAxis = zDir.unit();
   GlobalVector yAxis( zAxis.y(), -zAxis.x(), 0);
   GlobalVector xAxis = yAxis.cross( zAxis);
   return Surface::RotationType( xAxis, yAxis, zAxis);
   }
 */

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalEGMAnalysis);
