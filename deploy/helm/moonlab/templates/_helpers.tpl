{{/*
Standard labels applied to every resource in the chart.
*/}}
{{- define "moonlab.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels for a specific component (control, exporter, gateway).
*/}}
{{- define "moonlab.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Fully-qualified image name like "docker.io/moonlab/control-plane:1.0.4".
Takes a component name (control-plane / control-exporter / websocket-gateway).
*/}}
{{- define "moonlab.image" -}}
{{- printf "%s/%s/%s:%s"
      $.image.registry
      $.image.repository
      .component
      $.image.tag }}
{{- end }}
