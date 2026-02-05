{ self }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.style-search;
  serviceCfg = config.systemd.services.style-search;

  # Properties to pass through to systemd-run (exclude service-specific ones)
  excludedProps = [ "Type" "ExecStart" "ExecReload" "ExecStop" "Restart" "RestartSec" ];
  serviceConfigProps = lib.filterAttrs (k: v: !(builtins.elem k excludedProps)) serviceCfg.serviceConfig;

  # Admin CLI wrapper that runs commands in the same systemd context as the service
  adminScript = pkgs.writeShellScriptBin "style-search-admin" ''
    exec systemd-run \
      --pipe \
      --quiet \
      --wait \
      --collect \
      --service-type=exec \
      ${lib.concatStringsSep " \\\n      " (lib.mapAttrsToList (k: v: "--property=${k}=${toString v}") serviceConfigProps)} \
      ${lib.concatStringsSep " \\\n      " (lib.mapAttrsToList (k: v: "--property=Environment=${k}=${toString v}") serviceCfg.environment)} \
      ${cfg.package}/bin/style-api "$@"
  '';
in
{
  options.services.style-search = {
    enable = lib.mkEnableOption "style-search API server";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.system}.default;
      description = "The style-search package to use";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Host to bind to";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port to bind to";
    };

    dataDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/style-search";
      description = "Directory for data storage";
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ adminScript ];

    systemd.services.style-search = {
      description = "Style Search API Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/style-api serve -h ${cfg.host} -p ${toString cfg.port}";
        WorkingDirectory = cfg.dataDir;
        StateDirectory = "style-search";
        DynamicUser = true;
        Restart = "on-failure";
      };

      environment = {
        STYLE_SEARCH_DATA_DIR = cfg.dataDir;
      };
    };
  };
}
