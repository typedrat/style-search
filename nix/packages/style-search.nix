{
  lib,
  python3Packages,
  # bun2nix
  bun2nix,
}:
python3Packages.buildPythonApplication {
  pname = "style-search";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../..;

  nativeBuildInputs = [
    python3Packages.hatchling
    bun2nix.hook
  ];

  # bun2nix configuration
  bunRoot = "web";
  bunDeps = bun2nix.fetchBunDeps {
    bunNix = ../../web/bun.nix;
  };

  build-system = with python3Packages; [
    hatchling
  ];

  dependencies = with python3Packages; [
    beautifulsoup4
    chromadb
    click
    fastapi
    llvmlite
    numba
    numpy
    open-clip-torch
    pillow
    requests
    safetensors
    scikit-learn
    rich
    torch
    torchvision
    umap-learn
    uvicorn
    plotext
    textual
  ];

  preBuild = ''
    pushd web
    bun run build
    popd
  '';

  postInstall = ''
    mkdir -p $out/share/style-search
    cp -r web/dist $out/share/style-search/web
  '';

  # Set the static dir env var for the installed scripts
  makeWrapperArgs = [
    "--set STYLE_SEARCH_STATIC_DIR $out/share/style-search/web"
  ];

  meta = with lib; {
    description = "Visual style similarity search using CLIP embeddings";
    homepage = "https://github.com/awilliams/style-search";
    license = licenses.mit;
    platforms = platforms.linux ++ platforms.darwin;
    mainProgram = "style-api";
  };
}
