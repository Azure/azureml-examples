# Federated Learning for Contoso

## Contents

This page describes a plan for the third-party (3P) Contoso company/team/service to onboard to cross-silo _Federated Learning_ (FL) through the [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/#product-overview) service, while Azure ML itself matures. It is derived from a plan we are currently following with a first-party (1P) Microsoft team.

Even though Contoso's application described here is Federated Learning, the steps outlined in the plan are fairly generic. They are also applicable to the scenario where the data constraints and the need for _Federation_ are similar, but with no Machine Learning involved - _Federated Analysis_. 

## Introduction

Contoso runs state-of-the-art Machine Learning workloads to deliver insights from huge amounts of data. Contoso does not own the data - the _data providers_ do. Due to technical and legal constraints, some data cannot be moved to a central location. Contoso has decided to use Federated Learning to be able to train a single model that leverages all these distributed data. The data providers will provide access to their data, but only while the data stays in their own subscription, and Contoso will run FL code against it in their compute and storage silo. 

Considering Azure ML’s long-range plans of bringing federated learning to 3P, this document proposes a roadmap for how the Contoso team can build an MVP as quickly as possible, while minimizing throw-away work. The architecture fundamentals will remain consistent, but scaling Contoso out (more ML scenarios, extensibility author) will require the additional guarantees Azure ML will be bringing to the platform. 

## Plan

At a high level, the plan is for Contoso to run federated learning via a single, Contoso-owned workspace with attached customer-owned compute and storage. They will leverage the [in-development federated pipelines SDK](https://github.com/Azure/shrike/blob/main/docs/rfc/federated-learning.md) produced by the AIMS team to create and orchestrate these workloads. Azure ML will provide progressively stronger safety guarantees as its 3P-facing federated learning support lands. Before that support lands, some platform guarantees will have to be replaced with strict code review guarantees. 

Throughout this nine-month plan (divided into 3 quarters), 2 data scientists from the AIMS team, 2 researchers from MSR, and 2 data scientists from Contoso will contribute to the work. 

Rhythm of Business: need to agree on preferred way of communication, recurring meetings or not, work item tracking...

### Q1

AIMS & MSR teams onboard to Contoso's data and codebase. 
- **Milestone:** Folks in these teams can reproduce current Contoso jobs (whether they be training jobs or regular analyses). 

Bring AIMS & MSR folks into relevant security groups on the Contoso side, and likewise add Contoso to AIMS' contributors security group.
- **Milestone:** Contoso folks can see current synthetic federated pipelines, add everyone to "Federated Learning" Teams channel. 

AIMS & Contoso onboard to [FLUTE](https://github.com/microsoft/msrflute) for running their first simulations of federated jobs. Contoso can find more context at this page: aka.ms/aims/fl (internal only). 
- **Milestone:** Conduct ad-hoc eyes-on (public data) synthetic federated experiments & profiling, SKU profiling. These may leverage FLUTE. 

Decide on and write up detailed technical architecture document.
- **Milestone:** Customers sign off. 

Happy path: the data providers just give SAS keys + AML IP-protected compute in AIMS' subscription. Almost as happy: single AAD tenant and many subscriptions. Less happy but still possible: silos spread across several AAD tenants. 

AIMS to finish the "federated pipelines" SDK. 

Open questions

- Which compute target to use. Azure Arc-enabled Kubernetes vs. IP-protecting isolated compute? Fortunately, Azure Confidential Compute is not required.  
- Push to allow-list additional subscriptions for the 1P-ready policy service, or wait until AML ships this for 3P? 

Design for how this will scale horizontally to many data providers and scenarios with low trust. 

Be explicit about data protection / trust guarantees. 

Make sure forward-facing, minimize technical debt. 

**Milestones:** formal sign-off on architecture, successful synthetic experiments (reproduce current domain adaptation experiments in federated context), 3P resource architecture proposal to share with customers, documentation of customer setup. 

### Q2

Build and run 1P but fully representative version of federated pipelines. In other words, the data is Microsoft-owned and "safe", but everything else is as it will be. 

Build the "eyes-off" federated architecture (pointers to real customer data). 

Capture/document this process for Contoso's other scenarios. 

(Stretch) Explore a process for the providers to run multi-provider federated jobs as well.  

**Milestones:** Successful "eyes-on" experiments with real architecture, data access granted + build "eyes-off" architecture, sign-off on process for other scenarios. 

### Q3

Ship the federated Contoso pipelines to public preview with real customer data. 

Model training takes approximately one week of compute. 

Give buffer in case things slip. 

**Final Milestone:** Contoso can autonomously run their federated pipelines.