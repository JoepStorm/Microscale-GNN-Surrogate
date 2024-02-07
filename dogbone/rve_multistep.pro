userinput =
	  {
	    modules = [ "mesh", "pbcgroups"];
	  
	    mesh = 
	    {
	      type = "GmshInput";
	      file = "../dogbone/meshes/m_4.msh";
	      doElemGroups = true;
	    };
	  
	    pbcgroups = 
	    {
	      type = "PBCGroupInput";
	    };

	  };
	  
	  
	  model =
	  {
	    type        = "Matrix";
	    matrix.type = "Sparse";
	  
	    model       =
	    {
	      type   = "Multi";
	      models = [ "matrix"]; 
	  
	      matrix =
	      {
          type     = "Stress";
          elements = "gmsh0";
          writeStrains = true;
          writeStresses = true;

              material =
              {
                type   = "J2";
                rank   = 2;
                anmodel = "PLANE_STRESS"; // "PLANE_STRAIN"

                young = 3.13e3;
                poisson = 0.37;
                yield = "64.80-33.6*exp(x/-0.003407)";

                rmTolerance = 1e-10;
                rmMaxIter   = 1000;
              };


          shape.type = "Triangle3";
          shape.intScheme = "Gauss1";
	      };
	    };
	  };
	   
	  nonlin =
	  {
	    precision = 1.e-6;
	    solver.type = "SkylineLU";
	    solver.precon.type = "ILUd";
	    maxIter = 20;
	  };

