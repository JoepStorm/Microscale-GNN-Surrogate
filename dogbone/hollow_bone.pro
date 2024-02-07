timeStep = 7.5e-2;
minTimeStep = 1.0e-8;

log =
{
  pattern = "*.info";
  file    = "-$(CASE_NAME).log";
};

control	= 
{
  fgMode	= false;
  runWhile	= "i<25";
};

userinput =
{
  modules = [ "mesh", "groups"];

  mesh = 
  {
    type = "GmshInput";
    file = "../dogbone/mesh2.msh";
  };

  groups =
  {
    type = "GroupInput";
    nodeGroups = [ "right", "cornbr", "left", "cornbl" ];

    right.xtype = "max";
    cornbr.ytype = "min";
    cornbr.xtype = "max";
    left.xtype = "min";
    cornbl.xtype = "min";
    cornbl.ytype = "min";
  };
  
  
};


model =
{
  type        = "Matrix";
  matrix.type = "Sparse";

  model       =
  {
    type   = "Multi";
    models = [ "macro", "arclen" ];

    macro =
    {
      type     = "Stress";
      elements = "all";
      
      writeStrains = true;
      writeStresses = true;

      material =
      {
        type = "Observer";
              rank = 2;
        rseed = 23;
        shuffle = false;

        outFile = "../dogbone/sample.data";

        material =
        {
          type   = "FE2";
          rank   = 2;
          bcType = "periodic";
          rom    = false;
          samp   = false;
          output = false;
          debug  = false;

          maxSubStepLevel = 0;

          micro =
          {
            include "../dogbone/rve_multistep.pro";
          };
        };
      };
      
      shape.type = "Triangle3";
      shape.intScheme = "Gauss1";//*Gauss1";
    };

    arclen =
    {
      type = "BC";
//      mode = "disp";
      mode = "arclen";
      arclenGroup = 3; // not needed for mode = disp
      shape      = "t";
      step       = timeStep;

      nodeGroups = [ "right", "cornbr", "left", "cornbl" ];
      dofs       = [ "dx", "dy", "dx", "dy" ];
      unitVec    = [ 1.0, 0.0, 0.0, 0.0];
    };

  };
};
 
usermodules = 
{
  modules = [ "stepper" ];

  stepper = 
  {
    type = "AdaptiveStep";

    optIter = 5;
    reduction = 0.4;
    minIncr   = minTimeStep;
    maxIncr   = timeStep;
    startIncr = timeStep;

    solver = 
    {
//      type = "Nonlin";
      type = "Arclen";
      precision = 1.e-4;
      solver.type = "SkylineLU";
      maxIter = 20;
    };
  };
};
