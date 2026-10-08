#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
action="$repo_root/src/AscifyAction.cpp"
header="$repo_root/src/LocalHeader.h"
closure="$repo_root/src/LocalHeader.cpp"
main="$repo_root/src/main.cpp"
factory="$repo_root/src/ReplacementsFrontendActionFactory.h"

require_text() {
  needle=$1
  file=$2
  if ! grep -F -- "$needle" "$file" >/dev/null; then
    echo "local-header closure contract missing '$needle' in $file" >&2
    exit 1
  fi
}

require_absent() {
  needle=$1
  file=$2
  if grep -F -- "$needle" "$file" >/dev/null; then
    echo "local-header closure retained forbidden '$needle' in $file" >&2
    exit 1
  fi
}

require_text "LocalHeaderRewriteContext *localHeaderContext" "$factory"
require_text "localHeaderContext(context)" "$factory"
require_text "localHeaderContext" "$factory"
require_text "StringRef resolved_file_name" "$action"
require_text "file ? file->getName() : StringRef()" "$action"
require_text "const std::string resolvedPath = resolved_file_name.str()" "$action"
# This guards resolved include-path handling in PPCallbackProxy. Other code
# legitimately accesses FileEntryRef's underlying entry for physical identity.
inclusion_callback=$(sed -n '/^  void InclusionDirective(/,/^  void PragmaDirective(/p' "$action")
if [ -z "$inclusion_callback" ] ||
   printf '%s\n' "$inclusion_callback" | grep -F -- 'file->getFileEntry()' >/dev/null; then
  echo "local-header callback lost FileEntryRef include-path handling" >&2
  exit 1
fi
require_text "localHeaderContext->observe" "$action"
require_text "if (isAngled)" "$closure"
require_text "ascifySourceWithLocalHeaderClosure" "$main"
require_absent "ascifyLocalHeaders(" "$main"
require_text "node.sourcePath, pathString(stagedHeader)" "$closure"
require_text "ascify_exe, node.sourcePath" "$closure"
require_text ".ascify-local-closure" "$closure"
require_text "weaklyCanonical(rootArtifactPath)" "$closure"
require_text "dependencySources_.insert(canonical)" "$closure"
require_text "validateArtifactIsolation(inplace)" "$closure"
require_text "moveAside(rootArtifact, rootBackup, operationError)" "$closure"
require_text "reportRollback(rootArtifact, rootBackup, bundleArtifact" "$closure"
require_text "Rollback rename failed once; retrying" "$closure"
require_text "Published local-header closure but could not remove" "$closure"
require_text "Refusing to replace unowned local-header bundle" "$closure"
require_text "Refusing to retranslate a .dpp root" "$closure"
require_text "multiple inputs map to the same -o-dir artifact" "$main"
require_absent "LocalIncludeRe" "$closure"
require_absent "helper_cuda" "$closure"
# NVIDIA sample-helper closure is a separate provenance-gated frontend rule.
# Keep the recursive local-header engine free of helper special cases, and
# continue rejecting workload-name coupling in the frontend action.
require_absent "scan.cu" "$action"
require_absent "histogram64.cu" "$action"
require_absent "simpleAtomic" "$action"

printf '%s\n' "translated local include closure static contract passed"
