// SPDX-License-Identifier: GPL-2.0
// Author: Antoni Przybylik

#ifndef DATA_H_
#define DATA_H_

#include "neuron.h"

#define PROBES_CNT (81)

rl_t data[PROBES_CNT][3] = {
{0.000000000000000000000, 0.000000000000000000000, 0.519999999999999999983},
{0.000000000000000000000, 0.125000000000000000000, 0.412675547549528354529},
{0.000000000000000000000, 0.250000000000000000000, 0.148294880667081164043},
{0.000000000000000000000, 0.375000000000000000000, -0.131269449089095635564},
{0.000000000000000000000, 0.500000000000000000000, -0.275996998640178182902},
{0.000000000000000000000, 0.625000000000000000000, -0.208223742935824289040},
{0.000000000000000000000, 0.750000000000000000000, 0.035681680227688117610},
{0.000000000000000000000, 0.875000000000000000000, 0.324834190896736273266},
{0.000000000000000000000, 1.000000000000000000000, 0.504068114660146408194},
{0.125000000000000000000, 0.000000000000000000000, 0.842995215262429973852},
{0.125000000000000000000, 0.125000000000000000000, 0.735670762811958328371},
{0.125000000000000000000, 0.250000000000000000000, 0.471290095929511137926},
{0.125000000000000000000, 0.375000000000000000000, 0.191725766173334338305},
{0.125000000000000000000, 0.500000000000000000000, 0.046998216622251790961},
{0.125000000000000000000, 0.625000000000000000000, 0.114771472326605684836},
{0.125000000000000000000, 0.750000000000000000000, 0.358676895490118091486},
{0.125000000000000000000, 0.875000000000000000000, 0.647829406159166247135},
{0.125000000000000000000, 1.000000000000000000000, 0.827063329922576382063},
{0.250000000000000000000, 0.000000000000000000000, 0.947677428752103737607},
{0.250000000000000000000, 0.125000000000000000000, 0.840352976301632092180},
{0.250000000000000000000, 0.250000000000000000000, 0.575972309419184901654},
{0.250000000000000000000, 0.375000000000000000000, 0.296407979663008102033},
{0.250000000000000000000, 0.500000000000000000000, 0.151680430111925554682},
{0.250000000000000000000, 0.625000000000000000000, 0.219453685816279448558},
{0.250000000000000000000, 0.750000000000000000000, 0.463359108979791855214},
{0.250000000000000000000, 0.875000000000000000000, 0.752511619648840010890},
{0.250000000000000000000, 1.000000000000000000000, 0.931745543412250145873},
{0.375000000000000000000, 0.000000000000000000000, 0.772167089198219830262},
{0.375000000000000000000, 0.125000000000000000000, 0.664842636747748184781},
{0.375000000000000000000, 0.250000000000000000000, 0.400461969865300994282},
{0.375000000000000000000, 0.375000000000000000000, 0.120897640109124194695},
{0.375000000000000000000, 0.500000000000000000000, -0.023829909441958352649},
{0.375000000000000000000, 0.625000000000000000000, 0.043943346262395541223},
{0.375000000000000000000, 0.750000000000000000000, 0.287848769425907947869},
{0.375000000000000000000, 0.875000000000000000000, 0.577001280094956103491},
{0.375000000000000000000, 1.000000000000000000000, 0.756235203858366238474},
{0.500000000000000000000, 0.000000000000000000000, 0.457533261408167438394},
{0.500000000000000000000, 0.125000000000000000000, 0.350208808957695792914},
{0.500000000000000000000, 0.250000000000000000000, 0.085828142075248602421},
{0.500000000000000000000, 0.375000000000000000000, -0.193736187680928197179},
{0.500000000000000000000, 0.500000000000000000000, -0.338463737232010744544},
{0.500000000000000000000, 0.625000000000000000000, -0.270690481527656850628},
{0.500000000000000000000, 0.750000000000000000000, -0.026785058364144444004},
{0.500000000000000000000, 0.875000000000000000000, 0.262367452304903711624},
{0.500000000000000000000, 1.000000000000000000000, 0.441601376068313846579},
{0.625000000000000000000, 0.000000000000000000000, 0.246740637971481708272},
{0.625000000000000000000, 0.125000000000000000000, 0.139416185521010062805},
{0.625000000000000000000, 0.250000000000000000000, -0.124964481361437127688},
{0.625000000000000000000, 0.375000000000000000000, -0.404528811117613927302},
{0.625000000000000000000, 0.500000000000000000000, -0.549256360668696474666},
{0.625000000000000000000, 0.625000000000000000000, -0.481483104964342580750},
{0.625000000000000000000, 0.750000000000000000000, -0.237577681800830174135},
{0.625000000000000000000, 0.875000000000000000000, 0.051574828868217981528},
{0.625000000000000000000, 1.000000000000000000000, 0.230808752631628116484},
{0.750000000000000000000, 0.000000000000000000000, 0.310466204622884148082},
{0.750000000000000000000, 0.125000000000000000000, 0.203141752172412502615},
{0.750000000000000000000, 0.250000000000000000000, -0.061238914710034687881},
{0.750000000000000000000, 0.375000000000000000000, -0.340803244466211487492},
{0.750000000000000000000, 0.500000000000000000000, -0.485530794017294034829},
{0.750000000000000000000, 0.625000000000000000000, -0.417757538312940140967},
{0.750000000000000000000, 0.750000000000000000000, -0.173852115149427734325},
{0.750000000000000000000, 0.875000000000000000000, 0.115300395519620421332},
{0.750000000000000000000, 1.000000000000000000000, 0.294534319283030556294},
{0.875000000000000000000, 0.000000000000000000000, 0.624854462872794543763},
{0.875000000000000000000, 0.125000000000000000000, 0.517530010422322898282},
{0.875000000000000000000, 0.250000000000000000000, 0.253149343539875707756},
{0.875000000000000000000, 0.375000000000000000000, -0.026414986216301091838},
{0.875000000000000000000, 0.500000000000000000000, -0.171142535767383639176},
{0.875000000000000000000, 0.625000000000000000000, -0.103369280063029745307},
{0.875000000000000000000, 0.750000000000000000000, 0.140536143100482661342},
{0.875000000000000000000, 0.875000000000000000000, 0.429688653769530816992},
{0.875000000000000000000, 1.000000000000000000000, 0.608922577532940951920},
{1.000000000000000000000, 0.000000000000000000000, 0.988988458902601064442},
{1.000000000000000000000, 0.125000000000000000000, 0.881664006452129418907},
{1.000000000000000000000, 0.250000000000000000000, 0.617283339569682228489},
{1.000000000000000000000, 0.375000000000000000000, 0.337719009813505428868},
{1.000000000000000000000, 0.500000000000000000000, 0.192991460262422881517},
{1.000000000000000000000, 0.625000000000000000000, 0.260764715966776775393},
{1.000000000000000000000, 0.750000000000000000000, 0.504670139130289182022},
{1.000000000000000000000, 0.875000000000000000000, 0.793822649799337337671},
{1.000000000000000000000, 1.000000000000000000000, 0.973056573562747472599}};

#endif /* DATA_H_ */